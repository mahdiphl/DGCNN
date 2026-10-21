
import csv
import io
import json
import math
import shutil
import tarfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.datasets import ModelNet, ShapeNet
from torch_geometric.transforms import Compose, SamplePoints, NormalizeScale, FixedPoints
from complexity_score import compute_complexity_score

# 'shapenetpart' is the part-segmentation task: 50 global part labels (0..49)
# DONUT: genus 0..10 -> 11 classes; components (beta_0) 1..6 -> shifted to 0..5 -> 6 classes
NUM_CLASSES = {'modelnet40': 40, 'shapenet': 16, 'shapenetpart': 50,
               'donut_genus': 11, 'donut_components': 6}


class TranslatePointCloud:
    def __call__(self, data):
        xyz1 = torch.empty(3).uniform_(2./3., 3./2.)
        xyz2 = torch.empty(3).uniform_(-0.2, 0.2)
        data.pos = data.pos * xyz1 + xyz2
        return data


class ShufflePoints:
    """Shuffle point order. Every per-point attribute must be permuted with the
    same permutation, otherwise part-segmentation labels (data.y) would no
    longer line up with data.pos. Graph-level attributes (e.g. ModelNet's
    y of shape (1,)) are left alone by the size check."""
    def __call__(self, data):
        n = data.pos.size(0)
        perm = torch.randperm(n)
        for key in ('pos', 'y', 'x'):
            v = getattr(data, key, None)
            if torch.is_tensor(v) and v.dim() > 0 and v.size(0) == n:
                setattr(data, key, v[perm])
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


class ShapeNetPartPyG(ShapeNetPyG):
    """
    ShapeNet part SEGMENTATION. Returns
        (pointcloud (N,3), ect_features (N,), category (1,), part_labels (N,))
    part_labels are PyG's global part ids 0..49; category is 0..15 in the
    standard (alphabetical) ShapeNet-Part order.
    """
    def __getitem__(self, item):
        data = self.dataset[item]
        pointcloud = data.pos[:self.num_points]         # (N, 3)
        ect_features = compute_complexity_score(data)   # (N,)  must be aligned with data.pos
        category = data.category.view(-1)               # (1,)
        part_labels = data.y[:self.num_points].long()   # (N,)
        return pointcloud, ect_features, category, part_labels


class DonutPyG(Dataset):
    """
    DONUT (LouisM2001/donut): 8192-pt clouds of watertight manifolds with topological labels.
    Returns (pointcloud (N,3), ect_features (N,), label (1,)) like the other datasets.

    The Hub repo stores WebDataset shards (train/*.tar, test/*.tar, ~1 GB each incl. meshes,
    ~50 GB total). On first use this class streams the needed shards one by one, keeps ONLY
    the point clouds (float32 .npy) and labels, and deletes each shard after extraction:
        root/pcd/<id>.npy
        root/labels.csv          (id, split, shard, genus, components)
        root/shards.json         (cached shard list, so no network once downloaded)
        root/done_shards.txt     (resume / incremental download bookkeeping)
    Needs `pip install huggingface_hub`.

    fraction: use only the first ceil(fraction * n_shards) shards of EACH split (train and
        test separately). Shards were shuffled (seed 0) before sharding, so this is a random
        subset. Raising the fraction later only downloads the extra shards; lowering it
        reuses what is on disk. E.g. fraction=0.1 -> ~4 train + 1 test shards (~5 GB transfer).

    Partitions: 'train' -> train shards, anything else -> test shards (the dataset has no val).

    task: 'genus'      -> total genus, label = genus
          'components' -> number of connected components, label = components - 1
    """
    REPO_ID = 'LouisM2001/donut'
    LABEL_OFFSET = {'genus': 0, 'components': 1}
    LABEL_HEADER = ['id', 'split', 'shard', 'genus', 'components']

    def __init__(self, num_points, partition='train', task='genus', root='data/DONUT',
                 fraction=1.0):
        assert task in self.LABEL_OFFSET, f'Unknown DONUT task: {task}'
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f'fraction must be in (0, 1], got {fraction}')
        self.num_points = num_points
        self.task = task
        self.root = Path(root)
        self.fraction = fraction
        self.split = 'train' if partition == 'train' else 'test'
        self.transform = build_transform(num_points, partition == 'train', from_mesh=False)

        self.shards = self._select_shards()
        self._ensure_prepared()
        self.ids, self.labels = self._read_labels()

    # ------------------------------------------------------------ shard selection
    def _shard_list(self):
        """All train/test shard names in the repo (cached locally after the first call)."""
        cache = self.root / 'shards.json'
        if cache.exists():
            return json.loads(cache.read_text())
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:
            raise ImportError('DONUT auto-download needs `pip install huggingface_hub`.') from exc
        files = sorted(f for f in HfApi().list_repo_files(self.REPO_ID, repo_type='dataset')
                       if f.endswith('.tar') and f.split('/')[0] in ('train', 'test'))
        if not files:
            raise RuntimeError(f'No train/test .tar shards found in {self.REPO_ID}.')
        self.root.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(files))
        return files

    def _select_shards(self):
        all_shards = self._shard_list()
        selected = []
        for split in ('train', 'test'):
            split_shards = [s for s in all_shards if s.startswith(split + '/')]
            n = max(1, math.ceil(self.fraction * len(split_shards)))
            selected += split_shards[:n]
        return selected

    # ------------------------------------------------------- download / extract
    def _ensure_prepared(self):
        labels_path = self.root / 'labels.csv'
        done_path = self.root / 'done_shards.txt'

        # labels.csv from an older version of this class has no 'shard' column: start over
        if labels_path.exists() and labels_path.read_text().split('\n', 1)[0].split(',') \
                != self.LABEL_HEADER:
            print('DONUT: outdated labels.csv found, rebuilding it.')
            labels_path.unlink()
            done_path.unlink(missing_ok=True)

        done = set(done_path.read_text().split()) if done_path.exists() else set()
        todo = [s for s in self.shards if s not in done]
        if not todo:
            return

        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError('DONUT auto-download needs `pip install huggingface_hub`.') from exc

        print(f'DONUT: downloading {len(todo)} of {len(self.shards)} selected shards into '
              f'{self.root} (fraction={self.fraction}; ~1 GB transferred per shard; resumable).')
        pcd_dir = self.root / 'pcd'
        pcd_dir.mkdir(parents=True, exist_ok=True)
        tmp_dir = self.root / '_tmp'
        if not labels_path.exists():
            with labels_path.open('w', newline='') as f:
                csv.writer(f).writerow(self.LABEL_HEADER)

        for k, shard in enumerate(todo):
            print(f'  [{k + 1}/{len(todo)}] {shard}')
            tar_path = hf_hub_download(self.REPO_ID, shard, repo_type='dataset',
                                       local_dir=tmp_dir)
            rows = self._extract_shard(Path(tar_path), shard, pcd_dir)
            with labels_path.open('a', newline='') as f:
                csv.writer(f).writerows(rows)
            with done_path.open('a') as f:
                f.write(shard + '\n')
            Path(tar_path).unlink()          # free disk space right away

        shutil.rmtree(tmp_dir, ignore_errors=True)
        print('DONUT ready.')

    @staticmethod
    def _extract_shard(tar_path, shard, pcd_dir):
        """Save <key>.pcd.npy as float32 and collect labels from <key>.json."""
        split = shard.split('/')[0]
        pcd_suffix, json_suffix = '.pcd.npy', '.json'
        saved, meta = set(), {}
        with tarfile.open(tar_path, 'r|') as tar:     # sequential read, no random access
            for member in tar:
                if not member.isfile():
                    continue
                name = Path(member.name).name
                if name.endswith(pcd_suffix):
                    key = name[:-len(pcd_suffix)]
                    arr = np.load(io.BytesIO(tar.extractfile(member).read()))
                    np.save(pcd_dir / f'{key}.npy', arr.astype(np.float32))
                    saved.add(key)
                elif name.endswith(json_suffix):
                    key = name[:-len(json_suffix)]
                    meta[key] = json.loads(tar.extractfile(member).read())
        return [[k, split, shard, int(meta[k]['genus']), int(meta[k]['components'])]
                for k in sorted(saved & meta.keys())]

    # ------------------------------------------------------------------ labels
    def _read_labels(self):
        selected = set(self.shards)
        ids, labels = [], []
        with (self.root / 'labels.csv').open(newline='') as f:
            for row in csv.DictReader(f):
                if row['split'] == self.split and row['shard'] in selected:
                    ids.append(row['id'])
                    labels.append(int(row[self.task]) - self.LABEL_OFFSET[self.task])
        if not ids:
            raise RuntimeError(f"No '{self.split}' samples found in {self.root / 'labels.csv'}.")

        n_cls = NUM_CLASSES[f'donut_{self.task}']
        lo, hi = min(labels), max(labels)
        if lo < 0 or hi >= n_cls:
            raise ValueError(
                f"DONUT {self.task} labels span {lo}..{hi} but NUM_CLASSES['donut_{self.task}'] "
                f"= {n_cls}. Set it to {hi + 1} (and check LABEL_OFFSET).")
        return ids, labels

    # -------------------------------------------------------------------- data
    def __getitem__(self, item):
        pos = torch.from_numpy(np.load(self.root / 'pcd' / f'{self.ids[item]}.npy'))  # (8192, 3)
        data = self.transform(Data(pos=pos))                       # (num_points, 3)
        pointcloud = data.pos[:self.num_points]                    # (N, 3)
        ect_features = compute_complexity_score(data)              # (N,)
        label = torch.tensor([self.labels[item]], dtype=torch.long)  # (1,)
        return pointcloud, ect_features, label

    def __len__(self):
        return len(self.ids)


def get_dataset(name, partition, num_points, fraction=1.0):
    """fraction (0, 1] only applies to the DONUT datasets (fraction of shards to download/use)."""
    if name == 'modelnet40':
        return ModelNet40PyG(num_points=num_points, partition=partition)
    if name == 'shapenet':
        return ShapeNetPyG(num_points=num_points, partition=partition)
    if name == 'shapenetpart':
        return ShapeNetPartPyG(num_points=num_points, partition=partition)
    if name == 'donut_genus':
        return DonutPyG(num_points=num_points, partition=partition, task='genus',
                        fraction=fraction)
    if name == 'donut_components':
        return DonutPyG(num_points=num_points, partition=partition, task='components',
                        fraction=fraction)
    raise ValueError(f'Unknown dataset: {name}')
