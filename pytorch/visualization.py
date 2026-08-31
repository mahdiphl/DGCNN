"""
ECT Visualization for point cloud datasets.

Expected input shape: (N_nodes, NUM_THETAS * NUM_THETAS)
e.g. torch.Size([10000, 4096]) -> 10000 nodes, each with a 64x64 ECT matrix.

Usage:
    ect = compute_local_ect(dataset, NUM_THETAS=64)   # shape [10000, 4096]
    visualize_ect(ect, num_thetas=64)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def to_numpy(ect):
    """Accept torch.Tensor or np.ndarray, return np.ndarray float32."""
    if isinstance(ect, torch.Tensor):
        return ect.detach().cpu().numpy().astype(np.float32)
    return np.asarray(ect, dtype=np.float32)


def reshape_ect(ect_flat, num_thetas):
    """
    Reshape flat ECT vectors back to 2D matrices.

    Args:
        ect_flat : array of shape (N, num_thetas * num_thetas)
        num_thetas : int — the NUM_THETAS used during compute_local_ect

    Returns:
        array of shape (N, num_thetas, num_thetas)
        axis 0 : nodes
        axis 1 : theta directions
        axis 2 : bump/threshold steps (t)
    """
    N = ect_flat.shape[0]
    return ect_flat.reshape(N, num_thetas, num_thetas)


def direction_variance(ect_3d):
    """
    Variance of each theta column across threshold steps.
    Higher variance -> more topologically informative direction.

    Returns: (N, num_thetas) array
    """
    return ect_3d.var(axis=2)


def informative_mask(var_per_dir, percentile=60):
    """
    Boolean mask: True for directions whose variance exceeds `percentile`-th
    percentile (computed per node).

    Returns: (N, num_thetas) bool array
    """
    thresholds = np.percentile(var_per_dir, percentile, axis=1, keepdims=True)
    return var_per_dir >= thresholds


# ---------------------------------------------------------------------------
# Main visualization
# ---------------------------------------------------------------------------

def visualize_ect(
    ect,
    num_thetas=64,
    node_indices=None,
    n_cols=4,
    percentile=60,
    cmap_heatmap="RdBu_r",
    figsize_per_node=(4.5, 9),
    save_path=None,
    show=True,
):
    """
    Visualize ECT matrices for a selection of nodes.

    For each node this produces a 3-panel column:
      1. ECT heatmap  (theta x t)
      2. Direction variance profile  (informative directions highlighted)
      3. Informative-region mask overlay

    Args:
        ect          : torch.Tensor or np.ndarray, shape (N, num_thetas**2)
        num_thetas   : int  — must match the NUM_THETAS used at compute time
        node_indices : list[int] | None  — which nodes to plot; defaults to
                       [0, N//4, N//2, 3*N//4] (4 spread-out nodes)
        n_cols       : int  — max nodes per row
        percentile   : float  — variance percentile above which a direction
                       is considered "informative" (default 60)
        cmap_heatmap : str   — matplotlib colormap for the ECT heatmap
        figsize_per_node : tuple — (width, height) per node column
        save_path    : str | None — if given, figure is saved here
        show         : bool — whether to call plt.show()
    """
    ect_np = to_numpy(ect)
    N = ect_np.shape[0]

    if node_indices is None:
        node_indices = sorted(set([0, N // 4, N // 2, 3 * N // 4]))

    n_nodes = len(node_indices)
    n_rows_grid = (n_nodes + n_cols - 1) // n_cols   # rows of node-groups
    fw = figsize_per_node[0] * min(n_nodes, n_cols)
    fh = figsize_per_node[1] * n_rows_grid
    fig = plt.figure(figsize=(fw, fh))
    fig.patch.set_facecolor("#f7f6f2")

    ect_3d = reshape_ect(ect_np, num_thetas)          # (N, T, T)
    var_all = direction_variance(ect_3d)               # (N, T)
    mask_all = informative_mask(var_all, percentile)   # (N, T) bool

    theta_ticks = np.linspace(0, num_thetas - 1, 5)
    theta_labels = ["0°", "90°", "180°", "270°", "360°"]
    t_labels = ["0", "0.25", "0.5", "0.75", "1.0"]

    outer = gridspec.GridSpec(n_rows_grid, min(n_nodes, n_cols),
                              figure=fig, hspace=0.45, wspace=0.35)

    for plot_i, node_i in enumerate(node_indices):
        row = plot_i // n_cols
        col = plot_i % n_cols
        inner = gridspec.GridSpecFromSubplotSpec(
            3, 1, subplot_spec=outer[row, col], hspace=0.55
        )

        ect_mat = ect_3d[node_i]            # (T, T)  theta x t
        var_vec = var_all[node_i]           # (T,)
        mask_vec = mask_all[node_i]         # (T,) bool

        # ---- panel 1: ECT heatmap -----------------------------------------
        ax1 = fig.add_subplot(inner[0])
        vabs = max(abs(ect_mat.min()), abs(ect_mat.max())) + 1e-8
        norm = TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)
        im = ax1.imshow(
            ect_mat.T,           # plot as (t, theta) so theta is x-axis
            aspect="auto",
            origin="lower",
            cmap=cmap_heatmap,
            norm=norm,
        )
        plt.colorbar(im, ax=ax1, fraction=0.035, pad=0.03,
                     label="χ(K_t, θ)")
        ax1.set_title(f"Node {node_i}  — ECT heatmap", fontsize=9,
                      fontweight="bold", pad=4)
        ax1.set_xlabel("θ direction", fontsize=8)
        ax1.set_ylabel("threshold t", fontsize=8)
        ax1.set_xticks(theta_ticks)
        ax1.set_xticklabels(theta_labels, fontsize=7)
        ax1.set_yticks(np.linspace(0, num_thetas - 1, 5))
        ax1.set_yticklabels(t_labels, fontsize=7)

        # highlight informative theta columns with a semi-transparent overlay
        for ti, is_info in enumerate(mask_vec):
            if is_info:
                ax1.axvspan(ti - 0.5, ti + 0.5, color="#e34948", alpha=0.12,
                            linewidth=0)

        # ---- panel 2: direction variance profile --------------------------
        ax2 = fig.add_subplot(inner[1])
        thetas = np.linspace(0, 360, num_thetas, endpoint=False)
        ax2.fill_between(thetas, var_vec, alpha=0.18, color="#2a78d6")
        ax2.plot(thetas, var_vec, lw=1.2, color="#2a78d6", label="variance")

        thresh_val = np.percentile(var_vec, percentile)
        ax2.axhline(thresh_val, color="#eda100", lw=1, ls="--",
                    label=f"p{int(percentile)} threshold")

        # scatter informative peaks
        peak_mask = var_vec >= thresh_val
        ax2.scatter(thetas[peak_mask], var_vec[peak_mask],
                    s=18, color="#e34948", zorder=5, label="informative")

        ax2.set_title("Direction variance", fontsize=9, pad=4)
        ax2.set_xlabel("θ (degrees)", fontsize=8)
        ax2.set_ylabel("Var(χ)", fontsize=8)
        ax2.set_xlim(0, 360)
        ax2.set_xticks([0, 90, 180, 270, 360])
        ax2.tick_params(labelsize=7)
        ax2.legend(fontsize=6.5, loc="upper right", framealpha=0.7)
        ax2.set_facecolor("#f0efec")

        # ---- panel 3: informative region mask ----------------------------
        ax3 = fig.add_subplot(inner[2])
        masked = ect_mat.copy()
        # gray out low-variance directions
        gray_val = np.zeros_like(masked)
        display = np.where(mask_vec[:, None], masked, np.nan)

        # background gray layer
        bg = np.ones_like(ect_mat) * np.nan
        ax3.imshow(bg.T, aspect="auto", origin="lower",
                   cmap="Greys", vmin=0, vmax=1, alpha=0.0)
        ax3.imshow(
            np.where(~mask_vec[:, None], 0.5, np.nan).T,
            aspect="auto", origin="lower",
            cmap="Greys", vmin=0, vmax=1, alpha=0.35,
        )
        im3 = ax3.imshow(
            display.T,
            aspect="auto",
            origin="lower",
            cmap=cmap_heatmap,
            norm=norm,
        )
        plt.colorbar(im3, ax=ax3, fraction=0.035, pad=0.03, label="χ")

        n_info = mask_vec.sum()
        pct = 100 * n_info / num_thetas
        ax3.set_title(
            f"Informative mask  ({n_info}/{num_thetas} dirs, {pct:.0f}%)",
            fontsize=9, pad=4,
        )
        ax3.set_xlabel("θ direction", fontsize=8)
        ax3.set_ylabel("threshold t", fontsize=8)
        ax3.set_xticks(theta_ticks)
        ax3.set_xticklabels(theta_labels, fontsize=7)
        ax3.set_yticks(np.linspace(0, num_thetas - 1, 5))
        ax3.set_yticklabels(t_labels, fontsize=7)

    fig.suptitle(
        f"ECT Visualization  |  {N:,} nodes  ·  {num_thetas}×{num_thetas}  "
        f"·  informative threshold: p{int(percentile)} variance",
        fontsize=11, fontweight="bold", y=1.01,
    )

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"Saved to {save_path}")

    if show:
        plt.show()

    return fig


# ---------------------------------------------------------------------------
# Additional utility: global summary across all nodes
# ---------------------------------------------------------------------------

def visualize_ect_summary(
    ect,
    num_thetas=64,
    percentile=60,
    save_path=None,
    show=True,
):
    """
    Dataset-level ECT summary across all N nodes.

    Shows:
      1. Mean ECT heatmap (averaged over all nodes)
      2. Std-dev heatmap (where does the ECT vary across nodes?)
      3. Mean direction variance profile with std band
      4. Fraction of nodes for which each direction is "informative"

    Args:
        ect          : torch.Tensor or np.ndarray, shape (N, num_thetas**2)
        num_thetas   : int
        percentile   : float — variance percentile for informativeness
        save_path    : str | None
        show         : bool
    """
    ect_np = to_numpy(ect)
    N = ect_np.shape[0]
    ect_3d = reshape_ect(ect_np, num_thetas)     # (N, T, T)
    var_all = direction_variance(ect_3d)          # (N, T)
    mask_all = informative_mask(var_all, percentile)  # (N, T) bool

    mean_ect = ect_3d.mean(axis=0)   # (T, T)
    std_ect = ect_3d.std(axis=0)     # (T, T)
    mean_var = var_all.mean(axis=0)  # (T,)
    std_var = var_all.std(axis=0)
    info_frac = mask_all.mean(axis=0)  # fraction of nodes where informative

    thetas = np.linspace(0, 360, num_thetas, endpoint=False)
    theta_ticks = np.linspace(0, num_thetas - 1, 5)
    theta_labels = ["0°", "90°", "180°", "270°", "360°"]
    t_labels = ["0", "0.25", "0.5", "0.75", "1.0"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.patch.set_facecolor("#f7f6f2")
    fig.suptitle(
        f"ECT Summary  |  {N:,} nodes  ·  {num_thetas}×{num_thetas}",
        fontsize=13, fontweight="bold",
    )

    # 1. Mean ECT
    ax = axes[0, 0]
    vabs = max(abs(mean_ect.min()), abs(mean_ect.max())) + 1e-8
    norm = TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)
    im = ax.imshow(mean_ect.T, aspect="auto", origin="lower",
                   cmap="RdBu_r", norm=norm)
    plt.colorbar(im, ax=ax, label="mean χ(K_t, θ)")
    ax.set_title("Mean ECT (all nodes)", fontweight="bold")
    ax.set_xlabel("θ direction"); ax.set_ylabel("threshold t")
    ax.set_xticks(theta_ticks); ax.set_xticklabels(theta_labels)
    ax.set_yticks(np.linspace(0, num_thetas - 1, 5))
    ax.set_yticklabels(t_labels)

    # 2. Std-dev ECT
    ax = axes[0, 1]
    im2 = ax.imshow(std_ect.T, aspect="auto", origin="lower", cmap="YlOrRd")
    plt.colorbar(im2, ax=ax, label="std(χ) across nodes")
    ax.set_title("ECT variability across nodes", fontweight="bold")
    ax.set_xlabel("θ direction"); ax.set_ylabel("threshold t")
    ax.set_xticks(theta_ticks); ax.set_xticklabels(theta_labels)
    ax.set_yticks(np.linspace(0, num_thetas - 1, 5))
    ax.set_yticklabels(t_labels)

    # 3. Mean direction variance profile
    ax = axes[1, 0]
    ax.fill_between(thetas, mean_var - std_var, mean_var + std_var,
                    alpha=0.20, color="#2a78d6", label="±1 std")
    ax.plot(thetas, mean_var, lw=1.5, color="#2a78d6", label="mean variance")
    thresh = np.percentile(mean_var, percentile)
    ax.axhline(thresh, color="#eda100", lw=1.2, ls="--",
               label=f"p{int(percentile)} threshold")
    peak_mask = mean_var >= thresh
    ax.scatter(thetas[peak_mask], mean_var[peak_mask],
               s=20, color="#e34948", zorder=5, label="informative")
    ax.set_title("Mean direction variance profile", fontweight="bold")
    ax.set_xlabel("θ (degrees)"); ax.set_ylabel("Var(χ)")
    ax.set_xlim(0, 360); ax.set_xticks([0, 90, 180, 270, 360])
    ax.legend(fontsize=8); ax.set_facecolor("#f0efec")

    # 4. Fraction-informative per direction
    ax = axes[1, 1]
    ax.bar(thetas, info_frac * 100, width=360 / num_thetas,
           color=["#e34948" if f >= 0.5 else "#2a78d6" for f in info_frac],
           alpha=0.75, align="edge")
    ax.axhline(50, color="#eda100", lw=1.2, ls="--",
               label="50% of nodes")
    ax.set_title("% nodes where direction is informative", fontweight="bold")
    ax.set_xlabel("θ (degrees)"); ax.set_ylabel("% nodes")
    ax.set_xlim(0, 360); ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_ylim(0, 100); ax.legend(fontsize=8)
    ax.set_facecolor("#f0efec")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"Saved to {save_path}")

    if show:
        plt.show()

    return fig
