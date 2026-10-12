

import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature


# ---------------------------------------------------------------------------
# Nested block gating
# ---------------------------------------------------------------------------
#
# ECS (Edge Complexity Score) is now precomputed OUTSIDE the model, from the
# ECT representation of each point cloud (see compute_complexity in the
# dataset code) and normalized per-sample to [0, 1] there. The Dataset's
# __getitem__ is expected to load/attach this as a per-point tensor of shape
# (N,), which the DataLoader then batches into (B, N) alongside the point
# cloud tensor. The model receives it directly as an argument to forward()
# -- it does NOT recompute anything.
#
# Compatibility notes:
#   - DGCNN never downsamples points: every EdgeConv block operates on the
#     same N points, in the same order, as the input. So the same (B, N)
#     ECS tensor can be reused unchanged at whichever block the gate is
#     inserted after (conv1 / conv2 / conv3) -- no reindexing needed.
#   - Your compute_complexity() already min-max normalizes per sample
#     (over that sample's N nodes). NestedGate only does an additional
#     per-batch median subtraction on top of that (see below), which is
#     exactly the paper's design and composes cleanly with your
#     normalization -- no double-normalization conflict.
#   - A CPU/GPU-agnostic fallback (compute_ecs_score) is kept below purely
#     so the model is still runnable/testable when no precomputed score is
#     supplied (e.g. quick unit tests without the full data pipeline). It
#     is bypassed automatically whenever an external score is passed in.

def compute_ecs_score(x, k=20, idx=None):
    """
    Fallback / testing-only proxy ECS, computed from backbone features
    directly (mean distance to k nearest neighbors in feature space). Not
    used when an external, ECT-derived score is supplied to forward().

    x:  (B, C, N) backbone features at the point where the gate is inserted.
    Returns: (B, N) raw (un-normalized) per-point score.
    """
    batch_size = x.size(0)
    num_points = x.size(2)
    if idx is None:
        idx = knn(x, k=k)
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx_flat = (idx + idx_base).view(-1)

    _, num_dims, _ = x.size()
    x_t = x.transpose(2, 1).contiguous()                      # (B, N, C)
    neighbors = x_t.view(batch_size*num_points, -1)[idx_flat, :]
    neighbors = neighbors.view(batch_size, num_points, k, num_dims)
    center = x_t.view(batch_size, num_points, 1, num_dims)

    edge_vec = neighbors - center                              # (B, N, k, C)
    edge_len = edge_vec.norm(dim=-1)                            # (B, N, k)
    score = edge_len.mean(dim=-1)                                # (B, N)
    return score, idx


def normalize_score(score, eps=1e-6):
    """Per-batch-sample min-max normalization to [0, 1]. Only used by the
    internal fallback path above -- the external ECT-based score is already
    normalized by compute_complexity() before it ever reaches the model."""
    s_min = score.min(dim=1, keepdim=True)[0]
    s_max = score.max(dim=1, keepdim=True)[0]
    return (score - s_min) / (s_max - s_min + eps)


class NestedGate(nn.Module):
    """
    Computes the nested-block gate g_i in [0, 1] from a (normalized) ECS
    score, following:

        g_i = sigma( alpha * (s_i - med_B(s)) )     [deterministic, default]

        g_i = sigma( MLP([f_i ; s_i]) )              [learned alternative]

    med_B(s) is the per-batch median of the score, which re-centers the
    gate automatically under dataset-level shifts in complexity statistics
    (e.g. denser point clouds, different noise levels at test time).
    """
    def __init__(self, in_channels, alpha=4.0, learned=False):
        super(NestedGate, self).__init__()
        self.alpha = alpha
        self.learned = learned
        if learned:
            self.mlp = nn.Sequential(
                nn.Linear(in_channels + 1, in_channels // 2),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Linear(in_channels // 2, 1)
            )

    def forward(self, feat, score):
        """
        feat:  (B, C, N) backbone features at the gate's insertion point
               (only used by the learned variant).
        score: (B, N) normalized ECS score.
        Returns: (B, 1, N) gate values, broadcastable over channels.
        """
        if self.learned:
            f = feat.transpose(2, 1)                      # (B, N, C)
            s = score.unsqueeze(-1)                        # (B, N, 1)
            g = self.mlp(torch.cat([f, s], dim=-1))         # (B, N, 1)
            g = torch.sigmoid(g).transpose(2, 1)             # (B, 1, N)
        else:
            med = score.median(dim=1, keepdim=True)[0]        # (B, 1)  per-batch median
            g = torch.sigmoid(self.alpha * (score - med))       # (B, N)
            g = g.unsqueeze(1)                                     # (B, 1, N)
        return g


class PointNet(nn.Module):
    def __init__(self, args, output_channels=40):
        super(PointNet, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, args.emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        self.linear1 = nn.Linear(args.emb_dims, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Linear(512, output_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_max_pool1d(x, 1).squeeze()
        x = F.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)
        return x


class DGCNNBaseline(nn.Module):
    """
    Original, unmodified DGCNN -- kept only as a reference / for manual
    A-B comparison. main.py does NOT import this; it imports the gated
    `DGCNN` class below, which behaves identically to this one whenever
    it's called with complexity=None (args.use_gate=False).
    """
    def __init__(self, args, output_channels=40):
        super(DGCNNBaseline, self).__init__()
        self.args = args
        self.k = args.k

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x


class DGCNN(nn.Module):
    """
    DGCNN with nested block gating, matching the main.py contract:

        model = DGCNN(args, output_channels=40, gate_all_blocks=args.gate_all_blocks)
        logits = model(data, complexity=complexity)   # complexity: (B, N) in [0,1], or None

    Per block r (EC_r = conv1..conv4), when gating is active at that block:

        u_i^(r) = EC_r(f^(r-1))_i                          -> the block's normal output (x1/x2/x3/x4)
        v_i^(r) = EC_extra,r(u^(r))_i                        -> extra same-width EdgeConv, one per gated block
        f_i^(r) = (1 - g_i^(r)) u_i^(r) + g_i^(r) v_i^(r)      -> blended, replaces u^(r) going forward

    Design note: u^(r) and v^(r) must match channel-wise to blend elementwise.
    Since DGCNN's blocks increase channel width at every stage
    (64 -> 64 -> 128 -> 256), each gated block gets its OWN same-width extra
    EdgeConv + gate (64->64 for conv1/conv2, 128->128 for conv3, 256->256 for
    conv4), kept separate from the main channel-increasing path. Complex
    points get one extra same-resolution refinement pass at that block;
    simple points skip it (gate ~ 0). The rest of the DGCNN backbone is
    untouched.

    gate_all_blocks=False (default): gate only the first EdgeConv block (conv1).
    gate_all_blocks=True:            gate all four EdgeConv blocks (conv1-conv4),
                                      each with its own extra_conv + NestedGate.

    complexity: precomputed, per-point, already-normalized-to-[0,1] ECT-based
    score of shape (B, N), matching the point order of the input point cloud
    (see compute_complexity() in the dataset / get_complexity() in main.py).
    Since DGCNN keeps N constant across all EdgeConv layers, the SAME (B, N)
    tensor is reused unchanged at every gated block -- no reindexing needed.
    Pass complexity=None to run as the plain ungated baseline (all gated
    blocks are skipped entirely, output is identical to stock DGCNN).

    last_lambda: after a forward pass with complexity != None, holds a
    Python float -- the mean gate value over all points (and all gated
    blocks, if gate_all_blocks=True) from the most recent forward call.
    Stays None if complexity was None (no gating happened), matching
    main.py's `if lam is not None: ...` logging check.
    """
    def __init__(self, args, output_channels=40, gate_all_blocks=False,
                 learned_gate=False, gate_alpha=4.0):
        super(DGCNN, self).__init__()
        self.args = args
        self.k = args.k
        self.gate_all_blocks = gate_all_blocks
        # which blocks get a gate: just the first, or all four
        self.gated_blocks = ['conv1', 'conv2', 'conv3', 'conv4'] if gate_all_blocks else ['conv1']

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))

        # One extra same-width EdgeConv + gate per gated block. Channel
        # width per block: conv1/conv2 -> 64, conv3 -> 128, conv4 -> 256.
        block_channels = {'conv1': 64, 'conv2': 64, 'conv3': 128, 'conv4': 256}
        self.extra_convs = nn.ModuleDict()
        self.nested_gates = nn.ModuleDict()
        for block in self.gated_blocks:
            c = block_channels[block]
            self.extra_convs[block] = nn.Sequential(
                nn.Conv2d(c*2, c, kernel_size=1, bias=False),
                nn.BatchNorm2d(c),
                nn.LeakyReLU(negative_slope=0.2)
            )
            self.nested_gates[block] = NestedGate(in_channels=c,
                                                   alpha=gate_alpha,
                                                   learned=learned_gate)

        self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

        # Populated during forward() when complexity is not None; a Python
        # float (mean gate value across all points / all gated blocks used
        # in the last forward call), for logging (main.py reads this as
        # model.module.last_lambda). None whenever gating was skipped.
        self.last_lambda = None
        # Raw per-block gate tensors from the last forward call, kept for
        # deeper inspection (e.g. shape_level_gate_score below).
        self.last_gates = {}

    def _apply_nested_gate(self, block, u, complexity):
        """u: (B, C, N) block output ('u^(r)'). Returns the gated blend
        f^(r) = (1-g) u + g v, where v = EC_extra,block(u)."""
        v = get_graph_feature(u, k=self.k)
        v = self.extra_convs[block](v)
        v = v.max(dim=-1, keepdim=False)[0]              # (B, C, N), 'v^(r)'

        g = self.nested_gates[block](u, complexity)        # (B, 1, N)
        self.last_gates[block] = g.squeeze(1)                 # (B, N)

        blended = (1 - g) * u + g * v
        return blended

    def forward(self, x, complexity=None):
        """
        x:          (B, 3 or C_in, N) input point cloud.
        complexity: (B, N) precomputed ECT-based complexity score, already
                    normalized to [0, 1] (see compute_complexity() /
                    get_complexity() in main.py). Pass None to skip gating
                    entirely and run the plain ungated DGCNN.
        """
        batch_size = x.size(0)
        use_gate = complexity is not None
        if use_gate:
            self.last_gates = {}

        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]
        if use_gate and 'conv1' in self.gated_blocks:
            x1 = self._apply_nested_gate('conv1', x1, complexity)

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]
        if use_gate and 'conv2' in self.gated_blocks:
            x2 = self._apply_nested_gate('conv2', x2, complexity)

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]
        if use_gate and 'conv3' in self.gated_blocks:
            x3 = self._apply_nested_gate('conv3', x3, complexity)

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]
        if use_gate and 'conv4' in self.gated_blocks:
            x4 = self._apply_nested_gate('conv4', x4, complexity)

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        xa = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        xb = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((xa, xb), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)

        if use_gate and self.last_gates:
            self.last_lambda = torch.cat(
                [g.reshape(-1) for g in self.last_gates.values()]
            ).mean().item()
        else:
            self.last_lambda = None

        return x

    def shape_level_gate_score(self, block=None):
        """Pool the last computed per-point gates to a single shape-level
        control score per batch item (mean pooling), as described for the
        classification use case in the paper. If `block` is None, averages
        over all gated blocks from the last forward call."""
        if not self.last_gates:
            raise RuntimeError("Run a forward pass with complexity != None first.")
        if block is not None:
            return self.last_gates[block].mean(dim=1)          # (B,)
        stacked = torch.stack(list(self.last_gates.values()), dim=0)  # (num_blocks, B, N)
        return stacked.mean(dim=0).mean(dim=1)                            # (B,)


# ===========================================================================
# ADDED: segmentation model (bottom branch of the DGCNN paper's Fig. 3) with
# the same nested block gating as the classification DGCNN above.
# Everything above this line is the original nested_gating.py, unchanged.
# ===========================================================================
class TransformNet(nn.Module):
    """Spatial transform block: estimates a 3x3 matrix from the edge features
    (coords + neighbour differences) and applies it to the input points."""

    def __init__(self, k=20):
        super(TransformNet, self).__init__()
        self.k = k

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1, nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, bias=False),
                                   self.bn2, nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, kernel_size=1, bias=False),
                                   self.bn3, nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn4 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 256, bias=False)
        self.bn5 = nn.BatchNorm1d(256)

        self.transform = nn.Linear(256, 3 * 3)
        nn.init.constant_(self.transform.weight, 0)
        nn.init.eye_(self.transform.bias.view(3, 3))  # start at identity

    def forward(self, x):
        B = x.size(0)
        t = get_graph_feature(x, k=self.k)       # (B, 6, N, k)
        t = self.conv1(t)                        # (B, 64, N, k)
        t = self.conv2(t)                        # (B, 128, N, k)
        t = t.max(dim=-1, keepdim=False)[0]      # (B, 128, N)
        t = self.conv3(t)                        # (B, 1024, N)
        t = t.max(dim=-1, keepdim=False)[0]      # (B, 1024)

        t = F.leaky_relu(self.bn4(self.linear1(t)), negative_slope=0.2)
        t = F.leaky_relu(self.bn5(self.linear2(t)), negative_slope=0.2)
        t = self.transform(t).view(B, 3, 3)      # (B, 3, 3)

        return torch.bmm(x.transpose(2, 1), t).transpose(2, 1)  # (B, 3, N)


class DGCNN_Seg(nn.Module):
    """
    DGCNN part-segmentation network (Fig. 3, bottom branch) with nested
    block gating.

        point cloud (B,3,N) -> spatial transform
          -> EdgeConv mlp{64,64}  -> u1 -> [nested gate] -> x1   (64)
          -> EdgeConv mlp{64,64}  -> u2 -> [nested gate] -> x2   (64)
          -> EdgeConv mlp{64}     -> u3 -> [nested gate] -> x3   (64)
          -> concat(x1,x2,x3) (192) -> mlp{1024} -> max pool     (1024)
          -> concat mlp{64}(category one-hot)                    (1088)
          -> repeat N, concat (x1,x2,x3)                         (1280)
          -> mlp{256,256,128,p}                                  (B,p,N)

    Nested gate at block r (same as the classification DGCNN):
        u = EC_r(f^(r-1));  v = EC_extra,r(u);  g = NestedGate(u, complexity)
        f^(r) = (1-g) * u + g * v          (replaces u going forward, including
                                            in the skip concat for the head)

    gate_all_blocks=False (default): gate only EdgeConv 1.
    gate_all_blocks=True:            gate all three EdgeConv blocks, each with
                                     its own 64->64 extra EdgeConv + NestedGate.

    num_categories: size of the one-hot object-category vector (16 for
                    ShapeNet-Part); 0 for datasets without one (S3DIS).

    forward(x, label=None, complexity=None):
        x (B,3,N); label (B,num_categories) one-hot; complexity (B,N) in [0,1]
        or None for the ungated baseline.  Returns (B, num_classes, N).

    last_lambda / last_gates / shape_level_gate_score behave exactly as in
    the classification DGCNN.
    """

    def __init__(self, args, num_classes=50, num_categories=16,
                 gate_all_blocks=False, use_transform=True,
                 learned_gate=False, gate_alpha=4.0):
        super(DGCNN_Seg, self).__init__()
        self.args = args
        self.k = args.k
        self.num_categories = num_categories
        self.gate_all_blocks = gate_all_blocks
        self.gated_blocks = ['conv1', 'conv2', 'conv3'] if gate_all_blocks else ['conv1']

        self.transform_net = TransformNet(k=args.k) if use_transform else None

        # EdgeConv 1: mlp{64, 64}
        self.conv1 = nn.Sequential(
            nn.Conv2d(3 * 2, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64), nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64), nn.LeakyReLU(negative_slope=0.2))
        # EdgeConv 2: mlp{64, 64}
        self.conv2 = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64), nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64), nn.LeakyReLU(negative_slope=0.2))
        # EdgeConv 3: mlp{64}
        self.conv3 = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64), nn.LeakyReLU(negative_slope=0.2))

        # all three EdgeConv outputs are 64-wide -> one extra 64->64 EdgeConv
        # + NestedGate per gated block
        self.extra_convs = nn.ModuleDict()
        self.nested_gates = nn.ModuleDict()
        for block in self.gated_blocks:
            self.extra_convs[block] = nn.Sequential(
                nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(negative_slope=0.2))
            self.nested_gates[block] = NestedGate(in_channels=64,
                                                  alpha=gate_alpha,
                                                  learned=learned_gate)

        # mlp{1024} on concat(x1, x2, x3)
        self.conv4 = nn.Sequential(
            nn.Conv1d(192, args.emb_dims, kernel_size=1, bias=False),
            nn.BatchNorm1d(args.emb_dims), nn.LeakyReLU(negative_slope=0.2))

        # mlp{64} on the categorical vector
        if num_categories > 0:
            self.conv_cat = nn.Sequential(
                nn.Conv1d(num_categories, 64, kernel_size=1, bias=False),
                nn.BatchNorm1d(64), nn.LeakyReLU(negative_slope=0.2))
            head_in = args.emb_dims + 64 + 192   # 1280
        else:
            self.conv_cat = None
            head_in = args.emb_dims + 192

        # mlp{256, 256, 128, p}
        self.conv5 = nn.Sequential(
            nn.Conv1d(head_in, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256), nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv6 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256), nn.LeakyReLU(negative_slope=0.2))
        self.dp2 = nn.Dropout(p=args.dropout)
        self.conv7 = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128), nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Conv1d(128, num_classes, kernel_size=1, bias=True)

        self.last_lambda = None
        self.last_gates = {}

    def _apply_nested_gate(self, block, u, complexity):
        """f = (1-g) u + g v, with v = EC_extra,block(u)."""
        v = get_graph_feature(u, k=self.k)
        v = self.extra_convs[block](v)
        v = v.max(dim=-1, keepdim=False)[0]              # (B, 64, N)

        g = self.nested_gates[block](u, complexity)      # (B, 1, N)
        self.last_gates[block] = g.squeeze(1)            # (B, N)
        return (1 - g) * u + g * v

    def forward(self, x, label=None, complexity=None):
        B, _, N = x.size()
        use_gate = complexity is not None
        if use_gate:
            self.last_gates = {}

        if self.transform_net is not None:
            x = self.transform_net(x)

        feat = get_graph_feature(x, k=self.k)
        x1 = self.conv1(feat).max(dim=-1, keepdim=False)[0]          # (B, 64, N)
        if use_gate and 'conv1' in self.gated_blocks:
            x1 = self._apply_nested_gate('conv1', x1, complexity)

        feat = get_graph_feature(x1, k=self.k)
        x2 = self.conv2(feat).max(dim=-1, keepdim=False)[0]          # (B, 64, N)
        if use_gate and 'conv2' in self.gated_blocks:
            x2 = self._apply_nested_gate('conv2', x2, complexity)

        feat = get_graph_feature(x2, k=self.k)
        x3 = self.conv3(feat).max(dim=-1, keepdim=False)[0]          # (B, 64, N)
        if use_gate and 'conv3' in self.gated_blocks:
            x3 = self._apply_nested_gate('conv3', x3, complexity)

        local = torch.cat((x1, x2, x3), dim=1)                       # (B, 192, N)
        g_feat = self.conv4(local).max(dim=-1, keepdim=True)[0]      # (B, 1024, 1)

        if self.conv_cat is not None:
            if label is None:
                raise ValueError("label (B, num_categories) is required when "
                                 "num_categories > 0")
            cat = self.conv_cat(label.view(B, -1, 1))                # (B, 64, 1)
            g_feat = torch.cat((g_feat, cat), dim=1)                 # (B, 1088, 1)

        g_feat = g_feat.repeat(1, 1, N)
        feat = torch.cat((g_feat, x1, x2, x3), dim=1)                # (B, 1280, N)

        feat = self.dp1(self.conv5(feat))
        feat = self.dp2(self.conv6(feat))
        feat = self.conv7(feat)
        out = self.conv8(feat)                                       # (B, p, N)

        if use_gate and self.last_gates:
            self.last_lambda = torch.cat(
                [g.reshape(-1) for g in self.last_gates.values()]
            ).mean().item()
        else:
            self.last_lambda = None
        return out

    def shape_level_gate_score(self, block=None):
        """Mean-pooled gate per shape, (B,). Same as the classification model."""
        if not self.last_gates:
            raise RuntimeError("Run a forward pass with complexity != None first.")
        if block is not None:
            return self.last_gates[block].mean(dim=1)
        stacked = torch.stack(list(self.last_gates.values()), dim=0)
        return stacked.mean(dim=0).mean(dim=1)


def build_model(task, args, num_classes, num_categories=16,
                gate_all_blocks=False, use_transform=True,
                learned_gate=False, gate_alpha=4.0):
    """task: "cls" or "seg".  use_transform / num_categories: seg only."""
    task = task.lower()
    if task in ("cls", "classification"):
        return DGCNN(args, output_channels=num_classes,
                     gate_all_blocks=gate_all_blocks,
                     learned_gate=learned_gate, gate_alpha=gate_alpha)
    if task in ("seg", "segmentation"):
        return DGCNN_Seg(args, num_classes=num_classes,
                         num_categories=num_categories,
                         gate_all_blocks=gate_all_blocks,
                         use_transform=use_transform,
                         learned_gate=learned_gate, gate_alpha=gate_alpha)
    raise ValueError("unknown task '%s' (expected 'cls' or 'seg')" % task)


if __name__ == "__main__":
    class Args:
        k = 20
        emb_dims = 1024
        dropout = 0.5

    args = Args()
    B, N = 4, 1024
    x = torch.randn(B, 3, N)
    g = torch.rand(B, N)
    label = F.one_hot(torch.randint(0, 16, (B,)), 16).float()

    for all_blocks in (False, True):
        cls = build_model("cls", args, num_classes=40, gate_all_blocks=all_blocks)
        seg = build_model("seg", args, num_classes=50, gate_all_blocks=all_blocks)
        print("all_blocks=%s" % all_blocks)
        print("  cls plain:", tuple(cls(x).shape))
        print("  cls gated:", tuple(cls(x, complexity=g).shape), "mean gate =", cls.last_lambda)
        print("  seg plain:", tuple(seg(x, label).shape))
        print("  seg gated:", tuple(seg(x, label, complexity=g).shape), "mean gate =", seg.last_lambda)
