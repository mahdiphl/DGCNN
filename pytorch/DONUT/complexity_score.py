
import importlib
from pathlib import Path

import numpy as np
import torch

import local_ect
importlib.reload(local_ect)
from local_ect import compute_local_ect, make_directions
from torch_geometric.transforms import Compose, KNNGraph, NormalizeScale
from torch_geometric.data import Data

from visualize_ect_complexity import (
    to_numpy, reshape_ect, compute_complexity,
    fig_complexity_3d, fig_complexity_multiview,
    supp_ect_heatmaps, supp_complexity_histogram,
    make_dirs, MANUSCRIPT_DIR, SUPP_DIR,
    set_class_names, run_all_visualizations,
)


class PosToX:
    def __call__(self, data):
        data.x = data.pos
        del data.pos
        return data


ect_transform = Compose([
    KNNGraph(k=5),
    NormalizeScale(),
    PosToX(),
])


def to_data(obj):
    """Accept an (N, 3+) numpy array / torch tensor / PyG Data and return Data(pos=(N,3))."""
    if isinstance(obj, Data):
        pos = obj.pos if obj.pos is not None else obj.x
        y = getattr(obj, 'y', None)
    else:
        pos, y = obj, None
    pos = torch.as_tensor(np.asarray(pos) if not torch.is_tensor(pos) else pos)
    pos = pos[:, :3].float()          # drop normals / extra columns if present
    d = Data(pos=pos)
    if y is not None:
        d.y = y
    return d


def compute_complexity_score(data, method='entropy', num_thetas=64, v=None, node_chunk=None):
    """data: numpy array (N,3), torch tensor, or Data with .pos. Returns (N,) scores in [0,1]
    (min-max normalized over this one shape)."""
    data = ect_transform(to_data(data))
    ect = compute_local_ect(data, radius=1, ECT_TYPE='points',
                            NUM_THETAS=num_thetas, v=v, node_chunk=node_chunk)
    return compute_complexity(ect, num_thetas=num_thetas, method=method)


def load_npy_dataset(root, class_names=None, num_points=None, seed=0):
    """
    Expects root/<class_name>/**/*.npy, each file an (N, 3+) array.
    (If root has no class sub-folders, all .npy files go into one class.)
    Adapt this function if your layout differs.
    Returns (list[Data], class_names).
    """
    root = Path(root)
    if class_names is None:
        class_names = sorted(p.name for p in root.iterdir() if p.is_dir())
    rng = np.random.default_rng(seed)
    data_list = []
    if not class_names:
        class_names, groups = ['all'], [(0, sorted(root.rglob('*.npy')))]
    else:
        groups = [(y, sorted((root / c).rglob('*.npy'))) for y, c in enumerate(class_names)]
    for y, files in groups:
        for f in files:
            xyz = np.load(f)[:, :3]
            if num_points is not None and len(xyz) != num_points:
                idx = rng.choice(len(xyz), num_points, replace=len(xyz) < num_points)
                xyz = xyz[idx]
            d = to_data(xyz)
            d.y = torch.tensor([y])
            d.path = str(f)
            data_list.append(d)
    return data_list, class_names


def compute_ect_dataset(data_list, num_thetas=64, seed=0, node_chunk=None):
    """ECT for every shape with the SAME directions. Returns (total_nodes, num_thetas**2),
    concatenated in data_list order (what run_all_visualizations expects)."""
    v = make_directions(3, num_thetas, seed)
    out = []
    for d in data_list:
        d2 = ect_transform(to_data(d))
        out.append(compute_local_ect(d2, radius=1, ECT_TYPE='points',
                                     NUM_THETAS=num_thetas, v=v, node_chunk=node_chunk))
    return torch.cat(out, dim=0)


if __name__ == "__main__":
    # single file
    xyz = np.load('./data/donut/pcd/60827b0aa5b230c123a51060f0432557a9475701.npy')
    score = compute_complexity_score(xyz)
    print(score.shape, score.min(), score.max())

    # whole dataset + figures
    data_list, class_names = load_npy_dataset('./data/donut')
    # set_class_names(class_names)
    ect = compute_ect_dataset(data_list)
    run_all_visualizations(data_list, ect, num_thetas=64, method="entropy")
