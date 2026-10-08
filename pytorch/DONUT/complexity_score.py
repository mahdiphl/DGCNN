
import argparse
import importlib
import random
from pathlib import Path

import matplotlib.pyplot as plt
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

# ----------------------------------------------------------------------------
# Defaults (override from the command line, or edit here in a notebook)
# ----------------------------------------------------------------------------
PCD_DIR = "./data/donut/pcd"
NUM_VIS = 6          # <-- number of shapes to visualize
NUM_POINTS = None    # e.g. 2048 to subsample each cloud (faster ECT); None = keep all
NUM_THETAS = 64
METHOD = "entropy"
SEED = 0
OUT_DIR = "./figures_pcd"


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


def select_pcd_files(pcd_dir, num_vis, seed=0, first=False):
    """List *.npy files in pcd_dir and pick num_vis of them (random or first N)."""
    pcd_dir = Path(pcd_dir)
    files = sorted(pcd_dir.rglob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No .npy files found in {pcd_dir.resolve()}")
    k = min(num_vis, len(files))
    if k < num_vis:
        print(f"Only {len(files)} files available; visualizing {k}.")
    if first:
        return files[:k]
    return sorted(random.Random(seed).sample(files, k))


def load_pcd_files(files, num_points=None, seed=0):
    """Load point clouds from the given files into a list of PyG Data (pos=(N,3))."""
    rng = np.random.default_rng(seed)
    data_list = []
    for f in files:
        arr = np.load(f)
        if arr.ndim != 2 or arr.shape[1] < 3:
            print(f"Skipping {f.name}: unexpected shape {arr.shape}")
            continue
        xyz = arr[:, :3]
        if num_points is not None and len(xyz) != num_points:
            idx = rng.choice(len(xyz), num_points, replace=len(xyz) < num_points)
            xyz = xyz[idx]
        d = to_data(xyz)
        d.y = torch.tensor([0])
        d.path = str(f)
        data_list.append(d)
    return data_list


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


def plot_shapes(data_list, scores, out_dir, cols=3, show=True):
    """One 3D scatter per shape: left = plain shape, right = colored by complexity."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(data_list)
    rows = int(np.ceil(n / cols))

    fig = plt.figure(figsize=(5 * cols, 5 * rows))
    for i, (d, s) in enumerate(zip(data_list, scores)):
        pts = d.pos.numpy()
        s = np.asarray(to_numpy(s)).ravel()
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        sc = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=s, cmap="viridis",
                        s=3, vmin=0, vmax=1)
        ax.set_title(Path(d.path).name[:16] + "...", fontsize=9)
        ax.set_axis_off()
        # equal aspect so the shape isn't distorted
        ax.set_box_aspect(np.ptp(pts, axis=0) + 1e-9)
    fig.colorbar(sc, ax=fig.axes, shrink=0.6, label="complexity score")
    fig.savefig(out_dir / "shapes_complexity_grid.png", dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print(f"Saved grid to {out_dir / 'shapes_complexity_grid.png'}")


def main(pcd_dir=PCD_DIR, num_vis=NUM_VIS, num_points=NUM_POINTS,
         num_thetas=NUM_THETAS, method=METHOD, seed=SEED, first=False,
         out_dir=OUT_DIR, run_full_figures=True):
    files = select_pcd_files(pcd_dir, num_vis, seed=seed, first=first)
    print(f"Visualizing {len(files)} shapes from {pcd_dir}")
    data_list = load_pcd_files(files, num_points=num_points, seed=seed)

    # per-shape scores (same directions for all shapes)
    v = make_directions(3, num_thetas, seed)
    scores = [compute_complexity_score(d, method=method, num_thetas=num_thetas, v=v)
              for d in data_list]
    for d, s in zip(data_list, scores):
        s = np.asarray(to_numpy(s))
        print(f"{Path(d.path).name}: N={len(s)}, score min={s.min():.3f}, max={s.max():.3f}")

    plot_shapes(data_list, scores, out_dir)

    if run_full_figures:
        # your existing figure suite, restricted to the selected shapes
        ect = compute_ect_dataset(data_list, num_thetas=num_thetas, seed=seed)
        # set_class_names(['pcd'])
        run_all_visualizations(data_list, ect, num_thetas=num_thetas, method=method)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcd_dir", default=PCD_DIR)
    ap.add_argument("--num_vis", type=int, default=NUM_VIS, help="number of shapes to visualize")
    ap.add_argument("--num_points", type=int, default=NUM_POINTS)
    ap.add_argument("--num_thetas", type=int, default=NUM_THETAS)
    ap.add_argument("--method", default=METHOD)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--first", action="store_true", help="take the first N files instead of random")
    ap.add_argument("--out_dir", default=OUT_DIR)
    ap.add_argument("--skip_full_figures", action="store_true")
    args, _ = ap.parse_known_args()   # parse_known_args so it also works inside notebooks
    main(args.pcd_dir, args.num_vis, args.num_points, args.num_thetas, args.method,
         args.seed, args.first, args.out_dir, run_full_figures=not args.skip_full_figures)
