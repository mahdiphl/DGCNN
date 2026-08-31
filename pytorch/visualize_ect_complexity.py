"""
ECT Scalar Complexity Visualization for Point Cloud Datasets (ModelNet40/ShapeNet)
===================================================================================
Generates all figures for manuscript and supplementary material.

Scalar complexity score (normalized 0-1) is computed per node as the
area under the direction-variance curve of the ECT, then encoded as
color on the 3D point cloud.

OUTPUT FILES
------------
Manuscript (high-priority):
  fig1_complexity_3d_chair.png       — 3D point cloud colored by complexity (chair)
  fig2_complexity_3d_table.png       — 3D point cloud colored by complexity (table)
  fig3_complexity_multiview.png      — 4-view panel (front/side/top/iso) for one shape

Supplementary:
  supp1_ect_heatmaps.png             — ECT heatmaps for selected nodes
  supp2_ect_summary.png              — Dataset-level ECT summary
  supp3_complexity_histogram.png     — Distribution of complexity scores
  supp4_complexity_multiclass.png    — Complexity coloring across multiple shape classes
  supp5_complexity_vs_geometry.png   — Complexity score vs. local geometric properties

Usage
-----
    ect = compute_local_ect(dataset, NUM_THETAS=64)   # shape (N, 4096)
    run_all_visualizations(dataset, ect, num_thetas=64)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.colorbar import ColorbarBase
from mpl_toolkits.mplot3d import Axes3D
import torch
from torch_geometric.datasets import ModelNet
from torch_geometric.transforms import SamplePoints, NormalizeScale

# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------

MANUSCRIPT_DIR = "figures_manuscript"
SUPP_DIR = "figures_supplementary"

def make_dirs():
    os.makedirs(MANUSCRIPT_DIR, exist_ok=True)
    os.makedirs(SUPP_DIR, exist_ok=True)
    print(f"Output dirs: '{MANUSCRIPT_DIR}/'  and  '{SUPP_DIR}/'")


# ---------------------------------------------------------------------------
# Core utilities
# ---------------------------------------------------------------------------

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float32)
    return np.asarray(x, dtype=np.float32)


def reshape_ect(ect_flat, num_thetas):
    """(N, T*T) -> (N, T, T)   axis1=theta, axis2=t"""
    return to_numpy(ect_flat).reshape(-1, num_thetas, num_thetas)


def compute_complexity(ect_flat, num_thetas, method="variance_auc"):
    """
    Compute per-node scalar complexity score, normalized to [0, 1].

    Methods
    -------
    'variance_auc'  : area under the direction-variance curve  (recommended)
                      captures how much the ECT varies across directions
    'l2_norm'       : L2 norm of the full ECT vector
    'max_variance'  : peak directional variance
    'entropy'       : Shannon entropy of |ECT| values (normalized)
    """
    ect_3d = reshape_ect(ect_flat, num_thetas)         # (N, T, T)

    if method == "variance_auc":
        var_per_dir = ect_3d.var(axis=2)               # (N, T)
        scores = var_per_dir.mean(axis=1)              # (N,)

    elif method == "l2_norm":
        scores = np.linalg.norm(to_numpy(ect_flat), axis=1)

    elif method == "max_variance":
        var_per_dir = ect_3d.var(axis=2)
        scores = var_per_dir.max(axis=1)

    elif method == "entropy":
        arr = np.abs(to_numpy(ect_flat)) + 1e-8
        arr = arr / arr.sum(axis=1, keepdims=True)
        scores = -(arr * np.log(arr)).sum(axis=1)

    else:
        raise ValueError(f"Unknown method: {method}")

    # normalize to [0, 1]
    mn, mx = scores.min(), scores.max()
    return (scores - mn) / (mx - mn + 1e-10)


# ---------------------------------------------------------------------------
# Load a shape from ModelNet40
# ---------------------------------------------------------------------------

def load_shape(dataset, shape_idx=0, num_points=1024):
    """
    Extract XYZ coordinates for a single shape from a PyG dataset.

    Returns
    -------
    xyz : np.ndarray  (M, 3)
    label : int
    label_name : str
    """
    data = dataset[shape_idx]
    xyz = to_numpy(data.pos if hasattr(data, 'pos') and data.pos is not None
                   else data.x[:, :3])
    label = int(data.y.item()) if hasattr(data, 'y') else -1

    # ModelNet40 class names (ordered)
    CLASS_NAMES = [
        'airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car',
        'chair','cone','cup','curtain','desk','door','dresser','flower_pot',
        'glass_box','guitar','keyboard','lamp','laptop','mantel','monitor',
        'night_stand','person','piano','plant','radio','range_hood','sink',
        'sofa','stairs','stool','table','tent','toilet','tv_stand','vase',
        'wardrobe','xbox'
    ]
    label_name = CLASS_NAMES[label] if 0 <= label < len(CLASS_NAMES) else f"class_{label}"
    return xyz, label, label_name


def get_node_xyz(dataset, global_node_indices, shape_sizes):
    """
    Map global node indices (from compute_local_ect output) back to
    per-shape XYZ coordinates.

    Args:
        dataset          : PyG dataset
        global_node_indices : array (N,) of global node indices
        shape_sizes      : list of ints — number of nodes per shape

    Returns dict: shape_idx -> (local_xyz, local_complexity_indices)
    """
    boundaries = np.cumsum([0] + list(shape_sizes))
    result = {}
    for si in range(len(shape_sizes)):
        lo, hi = boundaries[si], boundaries[si + 1]
        mask = (global_node_indices >= lo) & (global_node_indices < hi)
        if mask.sum() == 0:
            continue
        result[si] = {
            "global_mask": mask,
            "local_indices": global_node_indices[mask] - lo,
        }
    return result


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

CMAP_COMPLEXITY = "plasma"   # perceptually uniform, prints well in greyscale

def _scatter3d(ax, xyz, scores, cmap=CMAP_COMPLEXITY, s=6, alpha=0.85,
               elev=20, azim=45, title=""):
    norm = Normalize(vmin=0, vmax=1)
    colors = cm.get_cmap(cmap)(norm(scores))
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2],
               c=colors, s=s, alpha=alpha, linewidths=0)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=9, pad=3)


def _add_colorbar(fig, ax, label="Complexity score (0–1)", cmap=CMAP_COMPLEXITY):
    sm = cm.ScalarMappable(cmap=cmap, norm=Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.04, shrink=0.7)
    cbar.set_label(label, fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    return cbar


# ---------------------------------------------------------------------------
# FIG 1 & 2  —  3D complexity coloring (manuscript)
# ---------------------------------------------------------------------------

def fig_complexity_3d(xyz, scores, label_name, save_path,
                      cmap=CMAP_COMPLEXITY, dpi=200):
    """
    Single-shape 3D scatter colored by complexity. Clean, manuscript-ready.
    """
    fig = plt.figure(figsize=(6, 5.5))
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(111, projection='3d')
    _scatter3d(ax, xyz, scores, cmap=cmap, s=8, alpha=0.9,
               elev=25, azim=45)
    _add_colorbar(fig, ax, label="ECT complexity (normalized)")
    fig.suptitle(
        f"ECT Scalar Complexity  —  {label_name.capitalize()}",
        fontsize=12, fontweight="bold", y=0.97
    )
    plt.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# FIG 3  —  4-view multiview panel (manuscript)
# ---------------------------------------------------------------------------

def fig_complexity_multiview(xyz, scores, label_name, save_path,
                              cmap=CMAP_COMPLEXITY, dpi=200):
    """
    Four viewpoints: isometric, front, side, top.
    """
    views = [
        ("Isometric",  25,  45),
        ("Front",       0,   0),
        ("Side",        0,  90),
        ("Top",        90,   0),
    ]
    fig = plt.figure(figsize=(14, 4))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        f"ECT Scalar Complexity — {label_name.capitalize()}  (4 views)",
        fontsize=12, fontweight="bold", y=1.01
    )
    axes = []
    for i, (view_name, elev, azim) in enumerate(views):
        ax = fig.add_subplot(1, 4, i + 1, projection='3d')
        _scatter3d(ax, xyz, scores, cmap=cmap, s=6, alpha=0.88,
                   elev=elev, azim=azim, title=view_name)
        axes.append(ax)

    _add_colorbar(fig, axes[-1], label="Complexity (0–1)")
    plt.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# SUPP 1  —  ECT heatmaps for selected nodes
# ---------------------------------------------------------------------------

def supp_ect_heatmaps(ect_flat, num_thetas, node_indices, save_path,
                      complexity_scores=None, dpi=150):
    ect_3d = reshape_ect(ect_flat, num_thetas)
    n = len(node_indices)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 7))
    fig.patch.set_facecolor("#f7f6f2")
    fig.suptitle("ECT Heatmaps and Direction Variance — Selected Nodes",
                 fontsize=12, fontweight="bold")

    theta_ticks = np.linspace(0, num_thetas - 1, 5)
    theta_labels = ["0°", "90°", "180°", "270°", "360°"]
    t_labels = ["0", "0.25", "0.5", "0.75", "1.0"]
    thetas = np.linspace(0, 360, num_thetas, endpoint=False)

    for col, ni in enumerate(node_indices):
        mat = ect_3d[ni]
        vabs = max(abs(mat.min()), abs(mat.max())) + 1e-8
        norm = TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)

        # heatmap
        ax = axes[0, col] if n > 1 else axes[0]
        im = ax.imshow(mat.T, aspect="auto", origin="lower",
                       cmap="RdBu_r", norm=norm)
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.03, label="χ")
        c_str = f"  |  complexity={complexity_scores[ni]:.3f}" \
                if complexity_scores is not None else ""
        ax.set_title(f"Node {ni}{c_str}", fontsize=8, fontweight="bold")
        ax.set_xlabel("θ direction", fontsize=8)
        ax.set_ylabel("threshold t", fontsize=8)
        ax.set_xticks(theta_ticks); ax.set_xticklabels(theta_labels, fontsize=7)
        ax.set_yticks(np.linspace(0, num_thetas - 1, 5))
        ax.set_yticklabels(t_labels, fontsize=7)

        # variance profile
        ax2 = axes[1, col] if n > 1 else axes[1]
        var_vec = mat.var(axis=1)
        ax2.fill_between(thetas, var_vec, alpha=0.18, color="#2a78d6")
        ax2.plot(thetas, var_vec, lw=1.3, color="#2a78d6")
        thresh = np.percentile(var_vec, 60)
        ax2.axhline(thresh, color="#eda100", lw=1, ls="--", label="p60")
        peak_mask = var_vec >= thresh
        ax2.scatter(thetas[peak_mask], var_vec[peak_mask],
                    s=14, color="#e34948", zorder=5)
        ax2.set_xlabel("θ (degrees)", fontsize=8)
        ax2.set_ylabel("Var(χ)", fontsize=8)
        ax2.set_xlim(0, 360); ax2.set_xticks([0, 90, 180, 270, 360])
        ax2.tick_params(labelsize=7)
        ax2.set_facecolor("#f0efec")

    plt.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# SUPP 2  —  Dataset-level ECT summary
# ---------------------------------------------------------------------------

def supp_ect_summary(ect_flat, num_thetas, save_path, dpi=150):
    ect_3d = reshape_ect(ect_flat, num_thetas)
    mean_ect = ect_3d.mean(axis=0)
    std_ect = ect_3d.std(axis=0)
    var_all = ect_3d.var(axis=2)
    mean_var = var_all.mean(axis=0)
    std_var = var_all.std(axis=0)
    thetas = np.linspace(0, 360, num_thetas, endpoint=False)
    theta_ticks = np.linspace(0, num_thetas - 1, 5)
    theta_labels = ["0°", "90°", "180°", "270°", "360°"]
    t_labels = ["0", "0.25", "0.5", "0.75", "1.0"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.patch.set_facecolor("#f7f6f2")
    fig.suptitle("ECT Dataset-Level Summary", fontsize=13, fontweight="bold")

    # mean ECT
    vabs = max(abs(mean_ect.min()), abs(mean_ect.max())) + 1e-8
    norm = TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)
    im = axes[0,0].imshow(mean_ect.T, aspect="auto", origin="lower",
                          cmap="RdBu_r", norm=norm)
    plt.colorbar(im, ax=axes[0,0], label="mean χ")
    axes[0,0].set_title("Mean ECT (all nodes)", fontweight="bold")
    axes[0,0].set_xlabel("θ direction"); axes[0,0].set_ylabel("threshold t")
    axes[0,0].set_xticks(theta_ticks); axes[0,0].set_xticklabels(theta_labels)
    axes[0,0].set_yticks(np.linspace(0, num_thetas-1, 5))
    axes[0,0].set_yticklabels(t_labels)

    # std ECT
    im2 = axes[0,1].imshow(std_ect.T, aspect="auto", origin="lower",
                            cmap="YlOrRd")
    plt.colorbar(im2, ax=axes[0,1], label="std(χ) across nodes")
    axes[0,1].set_title("ECT variability across nodes", fontweight="bold")
    axes[0,1].set_xlabel("θ direction"); axes[0,1].set_ylabel("threshold t")
    axes[0,1].set_xticks(theta_ticks); axes[0,1].set_xticklabels(theta_labels)
    axes[0,1].set_yticks(np.linspace(0, num_thetas-1, 5))
    axes[0,1].set_yticklabels(t_labels)

    # mean variance profile
    ax = axes[1,0]
    ax.fill_between(thetas, mean_var - std_var, mean_var + std_var,
                    alpha=0.20, color="#2a78d6", label="±1 std")
    ax.plot(thetas, mean_var, lw=1.5, color="#2a78d6", label="mean variance")
    thresh = np.percentile(mean_var, 60)
    ax.axhline(thresh, color="#eda100", lw=1.2, ls="--", label="p60 threshold")
    peak_mask = mean_var >= thresh
    ax.scatter(thetas[peak_mask], mean_var[peak_mask],
               s=20, color="#e34948", zorder=5, label="informative")
    ax.set_title("Mean direction variance profile", fontweight="bold")
    ax.set_xlabel("θ (degrees)"); ax.set_ylabel("Var(χ)")
    ax.set_xlim(0, 360); ax.set_xticks([0, 90, 180, 270, 360])
    ax.legend(fontsize=8); ax.set_facecolor("#f0efec")

    # informative fraction
    mask_all = var_all >= np.percentile(var_all, 60, axis=1, keepdims=True)
    info_frac = mask_all.mean(axis=0)
    ax = axes[1,1]
    ax.bar(thetas, info_frac * 100, width=360/num_thetas,
           color=["#e34948" if f >= 0.5 else "#2a78d6" for f in info_frac],
           alpha=0.75, align="edge")
    ax.axhline(50, color="#eda100", lw=1.2, ls="--", label="50% of nodes")
    ax.set_title("% nodes where direction is informative", fontweight="bold")
    ax.set_xlabel("θ (degrees)"); ax.set_ylabel("% nodes")
    ax.set_xlim(0, 360); ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_ylim(0, 100); ax.legend(fontsize=8); ax.set_facecolor("#f0efec")

    plt.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# SUPP 3  —  Complexity score histogram
# ---------------------------------------------------------------------------

def supp_complexity_histogram(complexity_scores, label_name, save_path,
                               method="variance_auc", dpi=150):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.patch.set_facecolor("#f7f6f2")
    fig.suptitle(
        f"Complexity Score Distribution — {label_name.capitalize()}"
        f"  (method: {method})",
        fontsize=11, fontweight="bold"
    )
    scores = complexity_scores

    # histogram
    ax = axes[0]
    n, bins, patches = ax.hist(scores, bins=50, edgecolor="none", alpha=0.85)
    # color bars by value
    norm = Normalize(vmin=0, vmax=1)
    for patch, left in zip(patches, bins[:-1]):
        patch.set_facecolor(cm.plasma(norm(left + 0.5 / 50)))
    ax.set_xlabel("Complexity score", fontsize=10)
    ax.set_ylabel("Node count", fontsize=10)
    ax.set_title("Histogram", fontsize=10)
    ax.set_facecolor("#f0efec")
    ax.axvline(scores.mean(), color="#e34948", lw=1.5, ls="--",
               label=f"mean = {scores.mean():.3f}")
    ax.axvline(np.median(scores), color="#eda100", lw=1.5, ls=":",
               label=f"median = {np.median(scores):.3f}")
    ax.legend(fontsize=8)

    # CDF
    ax2 = axes[1]
    sorted_s = np.sort(scores)
    cdf = np.arange(1, len(sorted_s) + 1) / len(sorted_s)
    ax2.plot(sorted_s, cdf * 100, lw=2, color="#2a78d6")
    ax2.fill_between(sorted_s, cdf * 100, alpha=0.10, color="#2a78d6")
    ax2.set_xlabel("Complexity score", fontsize=10)
    ax2.set_ylabel("Cumulative % of nodes", fontsize=10)
    ax2.set_title("Cumulative distribution", fontsize=10)
    ax2.set_facecolor("#f0efec")
    for pct in [25, 50, 75]:
        val = np.percentile(scores, pct)
        ax2.axvline(val, lw=1, ls="--", color="gray", alpha=0.6)
        ax2.text(val + 0.01, pct + 1, f"p{pct}={val:.2f}", fontsize=7, color="gray")

    plt.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# SUPP 4  —  Complexity coloring across multiple shape classes
# ---------------------------------------------------------------------------

def supp_multiclass_complexity(shapes_dict, save_path, dpi=150):
    """
    shapes_dict: {class_name: (xyz array (N,3), scores array (N,))}
    """
    classes = list(shapes_dict.keys())
    n = len(classes)
    fig = plt.figure(figsize=(4 * n, 5))
    fig.patch.set_facecolor("white")
    fig.suptitle("ECT Complexity Coloring Across Shape Classes",
                 fontsize=12, fontweight="bold")

    for i, cls in enumerate(classes):
        xyz, scores = shapes_dict[cls]
        ax = fig.add_subplot(1, n, i + 1, projection='3d')
        _scatter3d(ax, xyz, scores, s=6, alpha=0.88, elev=20, azim=45,
                   title=cls.capitalize())

    # shared colorbar on last axis
    _add_colorbar(fig, ax, label="Complexity (0–1)")
    plt.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# SUPP 5  —  Complexity vs. local geometry
# ---------------------------------------------------------------------------

def supp_complexity_vs_geometry(xyz, scores, save_path, dpi=150):
    """
    Scatter plots: complexity vs. local density, height (z), and
    distance from centroid.
    """
    centroid = xyz.mean(axis=0)
    dist_centroid = np.linalg.norm(xyz - centroid, axis=1)
    height = xyz[:, 2] - xyz[:, 2].min()
    height /= height.max() + 1e-8

    # local density: mean distance to 10 nearest neighbours (approx via sort)
    # for large N this is O(N^2) — subsample if needed
    MAX_N = 2000
    if len(xyz) > MAX_N:
        idx = np.random.default_rng(42).choice(len(xyz), MAX_N, replace=False)
        xyz_sub, s_sub = xyz[idx], scores[idx]
        dc_sub, h_sub = dist_centroid[idx], height[idx]
    else:
        xyz_sub, s_sub, dc_sub, h_sub = xyz, scores, dist_centroid, height

    diffs = xyz_sub[:, None, :] - xyz_sub[None, :, :]          # (M,M,3)
    dists = np.linalg.norm(diffs, axis=2)                       # (M,M)
    np.fill_diagonal(dists, np.inf)
    local_density = np.sort(dists, axis=1)[:, :10].mean(axis=1)
    local_density = (local_density - local_density.min()) / \
                    (local_density.max() - local_density.min() + 1e-8)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.patch.set_facecolor("#f7f6f2")
    fig.suptitle("Complexity Score vs. Local Geometric Properties",
                 fontsize=11, fontweight="bold")

    pairs = [
        (dc_sub, "Distance from centroid (norm.)", "#2a78d6"),
        (h_sub,  "Height (normalized z)",          "#5b8f3f"),
        (local_density, "Local density (mean 10-NN dist, norm.)", "#9b3fbf"),
    ]
    for ax, (geo, xlabel, color) in zip(axes, pairs):
        ax.scatter(geo, s_sub, s=4, alpha=0.25, color=color, linewidths=0)
        # trend line
        z = np.polyfit(geo, s_sub, 1)
        xr = np.linspace(geo.min(), geo.max(), 100)
        ax.plot(xr, np.polyval(z, xr), color="black", lw=1.5, ls="--",
                label=f"r={np.corrcoef(geo, s_sub)[0,1]:.3f}")
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("Complexity score", fontsize=9)
        ax.legend(fontsize=8)
        ax.set_facecolor("#f0efec")

    plt.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Master runner
# ---------------------------------------------------------------------------

def run_all_visualizations(
    dataset,
    ect_flat,
    num_thetas=64,
    method="variance_auc",
    shape_indices=None,
    node_sample_for_heatmap=None,
    dpi_manuscript=250,
    dpi_supp=150,
):
    """
    Generate and save all manuscript and supplementary figures.

    Args:
        dataset        : PyG dataset (ModelNet40 etc.)
        ect_flat       : torch.Tensor or np.ndarray, shape (N, num_thetas**2)
                         where N = total nodes across all shapes in dataset
        num_thetas     : int — must match compute_local_ect NUM_THETAS
        method         : complexity scoring method (see compute_complexity)
        shape_indices  : list of dataset indices to visualize; default [0, 1]
        node_sample_for_heatmap : list of global node indices for supp heatmap
        dpi_manuscript : DPI for manuscript figures (higher = larger file)
        dpi_supp       : DPI for supplementary figures
    """
    make_dirs()
    ect_np = to_numpy(ect_flat)
    N_total = ect_np.shape[0]

    print(f"\nComputing complexity scores ({method}) for {N_total:,} nodes...")
    complexity = compute_complexity(ect_np, num_thetas, method=method)
    print(f"  Score range: [{complexity.min():.4f}, {complexity.max():.4f}]"
          f"  mean={complexity.mean():.4f}")

    # figure out how many nodes per shape
    shape_sizes = [dataset[i].x.shape[0] for i in range(len(dataset))]

    if shape_indices is None:
        # find first chair and first table if possible
        CLASS_NAMES = [
            'airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car',
            'chair','cone','cup','curtain','desk','door','dresser','flower_pot',
            'glass_box','guitar','keyboard','lamp','laptop','mantel','monitor',
            'night_stand','person','piano','plant','radio','range_hood','sink',
            'sofa','stairs','stool','table','tent','toilet','tv_stand','vase',
            'wardrobe','xbox'
        ]
        chair_idx = next((i for i in range(len(dataset))
                          if CLASS_NAMES[int(dataset[i].y)] == 'chair'), 0)
        table_idx = next((i for i in range(len(dataset))
                          if CLASS_NAMES[int(dataset[i].y)] == 'table'), 1)
        shape_indices = [chair_idx, table_idx]

    # node offset mapping: shape i starts at sum(shape_sizes[:i])
    offsets = np.cumsum([0] + shape_sizes)

    print("\n--- MANUSCRIPT FIGURES ---")

    for fig_num, si in enumerate(shape_indices[:2], start=1):
        data = dataset[si]
        xyz = to_numpy(data.pos if hasattr(data, 'pos') and data.pos is not None
                       else data.x[:, :3])
        lo, hi = offsets[si], offsets[si + 1]
        node_scores = complexity[lo:hi]

        label = int(data.y.item())
        CLASS_NAMES = [
            'airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car',
            'chair','cone','cup','curtain','desk','door','dresser','flower_pot',
            'glass_box','guitar','keyboard','lamp','laptop','mantel','monitor',
            'night_stand','person','piano','plant','radio','range_hood','sink',
            'sofa','stairs','stool','table','tent','toilet','tv_stand','vase',
            'wardrobe','xbox'
        ]
        label_name = CLASS_NAMES[label] if 0 <= label < len(CLASS_NAMES) \
                     else f"shape_{si}"

        # Fig 1 / 2: single-view 3D
        path = os.path.join(MANUSCRIPT_DIR,
                            f"fig{fig_num}_complexity_3d_{label_name}.png")
        fig_complexity_3d(xyz, node_scores, label_name, path,
                          dpi=dpi_manuscript)

        # Fig 3: 4-view panel (for the first shape only)
        if fig_num == 1:
            path3 = os.path.join(MANUSCRIPT_DIR,
                                 "fig3_complexity_multiview.png")
            fig_complexity_multiview(xyz, node_scores, label_name, path3,
                                     dpi=dpi_manuscript)

    print("\n--- SUPPLEMENTARY FIGURES ---")

    # Supp 1: ECT heatmaps
    if node_sample_for_heatmap is None:
        node_sample_for_heatmap = [0, N_total // 4, N_total // 2,
                                    3 * N_total // 4]
    supp_ect_heatmaps(
        ect_np, num_thetas, node_sample_for_heatmap,
        save_path=os.path.join(SUPP_DIR, "supp1_ect_heatmaps.png"),
        complexity_scores=complexity, dpi=dpi_supp,
    )

    # Supp 2: dataset summary
    supp_ect_summary(
        ect_np, num_thetas,
        save_path=os.path.join(SUPP_DIR, "supp2_ect_summary.png"),
        dpi=dpi_supp,
    )

    # Supp 3: complexity histogram (using first shape)
    si0 = shape_indices[0]
    data0 = dataset[si0]
    label0 = CLASS_NAMES[int(data0.y.item())]
    lo0, hi0 = offsets[si0], offsets[si0 + 1]
    supp_complexity_histogram(
        complexity[lo0:hi0], label0,
        save_path=os.path.join(SUPP_DIR, "supp3_complexity_histogram.png"),
        method=method, dpi=dpi_supp,
    )

    # Supp 4: multi-class comparison
    shapes_dict = {}
    for si in shape_indices:
        data_i = dataset[si]
        xyz_i = to_numpy(data_i.pos if hasattr(data_i, 'pos')
                         and data_i.pos is not None else data_i.x[:, :3])
        li = CLASS_NAMES[int(data_i.y.item())]
        lo_i, hi_i = offsets[si], offsets[si + 1]
        shapes_dict[li] = (xyz_i, complexity[lo_i:hi_i])
    supp_multiclass_complexity(
        shapes_dict,
        save_path=os.path.join(SUPP_DIR, "supp4_complexity_multiclass.png"),
        dpi=dpi_supp,
    )

    # Supp 5: complexity vs. geometry
    xyz0 = to_numpy(data0.pos if hasattr(data0, 'pos')
                    and data0.pos is not None else data0.x[:, :3])
    supp_complexity_vs_geometry(
        xyz0, complexity[lo0:hi0],
        save_path=os.path.join(SUPP_DIR, "supp5_complexity_vs_geometry.png"),
        dpi=dpi_supp,
    )

    print(f"\nAll figures saved.")
    print(f"  Manuscript : {MANUSCRIPT_DIR}/  (3 files)")
    print(f"  Supplementary: {SUPP_DIR}/  (5 files)")


# ---------------------------------------------------------------------------
# Example / demo entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    print("Loading ModelNet10 (subset) for demonstration...")
    transform = SamplePoints(1024)
    dataset = ModelNet(root="/tmp/ModelNet10", name="10",
                       train=True, transform=transform)
    print(f"  Dataset: {len(dataset)} shapes")

    # Simulate ECT output: normally this comes from compute_local_ect()
    # Shape: (total_nodes_across_all_shapes, NUM_THETAS**2)
    NUM_THETAS = 64
    shape_sizes = [dataset[i].x.shape[0] for i in range(len(dataset))]
    N_total = sum(shape_sizes)

    print(f"  Total nodes: {N_total:,}  |  ECT dim: {NUM_THETAS**2}")
    print("  Generating synthetic ECT (replace with compute_local_ect output)...")

    rng = np.random.default_rng(42)
    ect_fake = rng.standard_normal((N_total, NUM_THETAS**2)).astype(np.float32)
    # inject structured signal: nodes in 90–270° range get higher values
    for i in range(N_total):
        freq = 1 + (i % 3)
        thetas_v = np.linspace(0, 2 * np.pi, NUM_THETAS)
        ts_v = np.linspace(0, 1, NUM_THETAS)
        signal = np.sin(freq * thetas_v)[:, None] * np.cos(np.pi * ts_v)[None, :]
        ect_fake[i] += 0.8 * signal.ravel()

    # run everything
    run_all_visualizations(
        dataset=dataset,
        ect_flat=ect_fake,
        num_thetas=NUM_THETAS,
        method="variance_auc",
        dpi_manuscript=250,
        dpi_supp=150,
    )
