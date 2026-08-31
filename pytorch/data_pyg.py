
import torch
from torch.utils.data import Dataset
from torch_geometric.datasets import ModelNet
from torch_geometric.transforms import Compose, SamplePoints, NormalizeScale, KNNGraph, FaceToEdge
from complexity_score import compute_complexity_score

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

train_transform = Compose([
    SamplePoints(num=1024),   # set from args.num_points
    NormalizeScale(),
    TranslatePointCloud(),
    ShufflePoints(),
])

test_transform = Compose([
    SamplePoints(num=1024),
    NormalizeScale(),
])


class ModelNet40PyG(Dataset):
    """Wraps torch_geometric ModelNet, returns (pointcloud, label) like the old ModelNet40."""
    def __init__(self, num_points, partition='train'):
        transform = train_transform if partition == 'train' else test_transform
        self.dataset = ModelNet(root='data/ModelNet40', name='40',
                                 train=(partition == 'train'), transform=transform)
        self.num_points = num_points

    def __getitem__(self, item):
        data = self.dataset[item]
        pointcloud = data.pos[:self.num_points]# (N, 3) float32
        ect_features = compute_complexity_score(data)
        label = data.y                                  # scalar tensor
        return pointcloud, ect_features, label

    def __len__(self):
        return len(self.dataset)
