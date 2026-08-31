

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DGCNN + ECT complexity gating.

g_i = per-node complexity score from compute_complexity_score() (ECT-derived,
normalized to [0, 1]). Injected via:

    f_tilde_i = f_i * (1 + lambda * g_i)

applied to point features entering an EdgeConv block. Default: first block
only. Set gate_all_blocks=True to inject at every block instead.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (B, N, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature


class ComplexityGate(nn.Module):
    """
    f_tilde_i = f_i * (1 + lambda * g_i)

    x: (B, C, N) point features
    g: (B, N)    per-node ECT complexity score, in [0, 1]

    lambda is a single learnable scalar (shared across whichever blocks it's
    applied to), initialized at 0.5. At g=0 or lambda=0 this is the identity,
    so the gate can only choose to use the signal, never actively hurt the
    baseline model by construction.
    """

    def __init__(self, init_lambda: float = 0.5):
        super().__init__()
        self.lam = nn.Parameter(torch.tensor(float(init_lambda)))

    def forward(self, x, g):
        g = g.unsqueeze(1)  # (B, N) -> (B, 1, N), broadcasts over channels
        return x * (1.0 + self.lam * g)


class DGCNN(nn.Module):
    def __init__(self, args, output_channels=40, gate_all_blocks: bool = False):
        super().__init__()
        self.args = args
        self.k = args.k
        self.gate_all_blocks = gate_all_blocks

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                    self.bn1, nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                    self.bn2, nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                    self.bn3, nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
                                    self.bn4, nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                    self.bn5, nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(args.emb_dims * 2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

        # single shared gate; reused at every gated block if gate_all_blocks=True
        self.gate = ComplexityGate(init_lambda=0.5)

        # diagnostic: read this after each step/epoch for the "did the
        # network reject the signal" check (lambda -> 0 means yes)
        self.last_lambda = None

    def forward(self, x, complexity=None):
        """
        x: (B, 3, N) input point cloud
        complexity: (B, N) ECT complexity score in [0, 1], or None to run
                    the model ungated (plain DGCNN baseline)
        """
        batch_size = x.size(0)
        use_gate = complexity is not None

        if use_gate:
            x = self.gate(x, complexity)
            self.last_lambda = self.gate.lam.detach().item()

        feat = get_graph_feature(x, k=self.k)
        feat = self.conv1(feat)
        x1 = feat.max(dim=-1, keepdim=False)[0]

        if use_gate and self.gate_all_blocks:
            x1 = self.gate(x1, complexity)
        feat = get_graph_feature(x1, k=self.k)
        feat = self.conv2(feat)
        x2 = feat.max(dim=-1, keepdim=False)[0]

        if use_gate and self.gate_all_blocks:
            x2 = self.gate(x2, complexity)
        feat = get_graph_feature(x2, k=self.k)
        feat = self.conv3(feat)
        x3 = feat.max(dim=-1, keepdim=False)[0]

        if use_gate and self.gate_all_blocks:
            x3 = self.gate(x3, complexity)
        feat = get_graph_feature(x3, k=self.k)
        feat = self.conv4(feat)
        x4 = feat.max(dim=-1, keepdim=False)[0]

        feat = torch.cat((x1, x2, x3, x4), dim=1)
        feat = self.conv5(feat)
        p1 = F.adaptive_max_pool1d(feat, 1).view(batch_size, -1)
        p2 = F.adaptive_avg_pool1d(feat, 1).view(batch_size, -1)
        feat = torch.cat((p1, p2), 1)

        feat = F.leaky_relu(self.bn6(self.linear1(feat)), negative_slope=0.2)
        feat = self.dp1(feat)
        feat = F.leaky_relu(self.bn7(self.linear2(feat)), negative_slope=0.2)
        feat = self.dp2(feat)
        out = self.linear3(feat)
        return out


if __name__ == "__main__":
    class Args:
        k = 20
        emb_dims = 1024
        dropout = 0.5

    args = Args()
    B, N = 4, 1024
    x = torch.randn(B, 3, N)
    complexity = torch.rand(B, N)  # stand-in for compute_complexity_score() output

    baseline = DGCNN(args, output_channels=40)
    y_plain = baseline(x)  # complexity=None -> ungated baseline forward pass
    print("baseline (no gate): ", tuple(y_plain.shape))

    gated_first = DGCNN(args, output_channels=40, gate_all_blocks=False)
    y1 = gated_first(x, complexity=complexity)
    print("gated, first block only:", tuple(y1.shape), "lambda =", gated_first.last_lambda)

    gated_all = DGCNN(args, output_channels=40, gate_all_blocks=True)
    y2 = gated_all(x, complexity=complexity)
    print("gated, all blocks:      ", tuple(y2.shape), "lambda =", gated_all.last_lambda)
