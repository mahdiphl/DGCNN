
import torch
from torch.utils.data import Dataset
from torch_geometric.datasets import ModelNet, ShapeNet
from torch_geometric.transforms import Compose, SamplePoints, NormalizeScale, FixedPoints
from complexity_score import compute_complexity_score

NUM_CLASSES = {'modelnet40': 40, 'shapenet': 16}


class TranslatePointCloud:
    def __call__(self, data):
        xyz1 = torch.empty(3).uniform_(2./3., 3./2.)
        xyz2 = torch.empty(3).uniform_(-0.2, 0.2)
        data.pos = data.pos * xyz1 + xyz2
        return data


class ShufflePoints:
    def __call__(self, data):
        perm = torch.randperm(data.pos.size(0))
        data.pos = data.pos[perm]
        return data


def build_transform(num_points, train, from_mesh):
    """
    from_mesh=True  -> ModelNet (meshes): sample points on faces.
    from_mesh=False -> ShapeNet (already point clouds, variable size ~2-3k pts):
                       subsample/pad to a fixed number so batches can be stacked.
    """
    sampler = SamplePoints(num=num_points) if from_mesh else FixedPoints(num=num_points, replace=False,
                                                                         allow_duplicates=True)
    steps = [sampler, NormalizeScale()]
    if train:
        steps += [TranslatePointCloud(), ShufflePoints()]
    return Compose(steps)


class ModelNet40PyG(Dataset):
    """Wraps torch_geometric ModelNet, returns (pointcloud, ect_features, label)."""
    def __init__(self, num_points, partition='train'):
        transform = build_transform(num_points, partition == 'train', from_mesh=True)
        self.dataset = ModelNet(root='data/ModelNet40', name='40',
                                train=(partition == 'train'), transform=transform)
        self.num_points = num_points

    def __getitem__(self, item):
        data = self.dataset[item]
        pointcloud = data.pos[:self.num_points]         # (N, 3)
        ect_features = compute_complexity_score(data)   # (N,)
        label = data.y                                  # (1,)
        return pointcloud, ect_features, label

    def __len__(self):
        return len(self.dataset)


class ShapeNetPyG(Dataset):
    """
    ShapeNet (16-category part-segmentation benchmark version from PyG) used for
    shape CLASSIFICATION: label = object category (0..15), not the per-point part labels.
    """
    def __init__(self, num_points, partition='train'):
        # 'trainval' for training (common practice), 'test' for evaluation
        split = 'trainval' if partition == 'train' else 'test'
        transform = build_transform(num_points, partition == 'train', from_mesh=False)
        self.dataset = ShapeNet(root='data/ShapeNet', categories=None,  # None = all 16 categories
                                include_normals=False, split=split, transform=transform)
        self.num_points = num_points

    def __getitem__(self, item):
        data = self.dataset[item]
        pointcloud = data.pos[:self.num_points]         # (N, 3)
        ect_features = compute_complexity_score(data)   # (N,)
        label = data.category.view(-1)                  # (1,) object category
        return pointcloud, ect_features, label

    def __len__(self):
        return len(self.dataset)


def get_dataset(name, partition, num_points):
    if name == 'modelnet40':
        return ModelNet40PyG(num_points=num_points, partition=partition)
    if name == 'shapenet':
        return ShapeNetPyG(num_points=num_points, partition=partition)
    raise ValueError(f'Unknown dataset: {name}')
