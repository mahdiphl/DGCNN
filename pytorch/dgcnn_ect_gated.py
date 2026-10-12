

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DGCNN + ECT complexity gating: classification AND segmentation in one module.

    from dgcnn_ect_gated_models import DGCNN, DGCNN_Seg, build_model

    model = build_model("cls", args, num_classes=40)
    model = build_model("seg", args, num_classes=50, num_categories=16)

Gate (shared definition for both models):

    f_tilde_i = f_i * (1 + lambda * g_i)

g: (B, N) per-node ECT complexity score in [0, 1] (or None -> plain baseline).

Forward signatures
    DGCNN     (classification): model(x, complexity=None)            -> (B, num_classes)
    DGCNN_Seg (segmentation)  : model(x, label=None, complexity=None) -> (B, num_classes, N)
    x: (B, 3, N)

Both models expose `last_lambda` (float, set after each gated forward pass)
for the "did the network reject the signal" diagnostic.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================================================
# shared helpers
# ==========================================================================
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

    lambda is a single learnable scalar (shared across gated blocks),
    initialised at 0.5. At g=0 or lambda=0 this is the identity.
    """

    def __init__(self, init_lambda: float = 0.5):
        super().__init__()
        self.lam = nn.Parameter(torch.tensor(float(init_lambda)))

    def forward(self, x, g):
        g = g.unsqueeze(1)  # (B, N) -> (B, 1, N)
        return x * (1.0 + self.lam * g)


class TransformNet(nn.Module):
    """Spatial transform block: estimates a 3x3 matrix from the edge features
    (coords + neighbour differences) and applies it to the input points."""

    def __init__(self, k=20):
        super().__init__()
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


# ==========================================================================
# classification (unchanged from your implementation)
# ==========================================================================
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

        self.gate = ComplexityGate(init_lambda=0.5)
        self.last_lambda = None

    def forward(self, x, complexity=None):
        """
        x: (B, 3, N); complexity: (B, N) in [0, 1] or None (ungated baseline)
        returns: (B, output_channels)
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
        return self.linear3(feat)


# ==========================================================================
# segmentation (bottom branch of Fig. 3)
# ==========================================================================
class DGCNN_Seg(nn.Module):
    """
    num_classes    : p, number of per-point semantic labels
    num_categories : size of the one-hot object-category vector (16 for
                     ShapeNet-Part). Use 0 for datasets without one (S3DIS).
    """

    def __init__(self, args, num_classes=50, num_categories=16,
                 gate_all_blocks: bool = False, use_transform: bool = True):
        super().__init__()
        self.args = args
        self.k = args.k
        self.num_categories = num_categories
        self.gate_all_blocks = gate_all_blocks

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

        # mlp{1024} on concat(x1, x2, x3) -> global descriptor
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

        self.gate = ComplexityGate(init_lambda=0.5)
        self.last_lambda = None

    def forward(self, x, label=None, complexity=None):
        """
        x: (B, 3, N); label: (B, num_categories) one-hot or None;
        complexity: (B, N) in [0, 1] or None (ungated baseline)
        returns: (B, num_classes, N)
        """
        B, _, N = x.size()
        use_gate = complexity is not None

        if self.transform_net is not None:
            x = self.transform_net(x)

        if use_gate:
            x = self.gate(x, complexity)
            self.last_lambda = self.gate.lam.detach().item()

        feat = get_graph_feature(x, k=self.k)
        x1 = self.conv1(feat).max(dim=-1)[0]            # (B, 64, N)

        x1_in = self.gate(x1, complexity) if (use_gate and self.gate_all_blocks) else x1
        feat = get_graph_feature(x1_in, k=self.k)
        x2 = self.conv2(feat).max(dim=-1)[0]            # (B, 64, N)

        x2_in = self.gate(x2, complexity) if (use_gate and self.gate_all_blocks) else x2
        feat = get_graph_feature(x2_in, k=self.k)
        x3 = self.conv3(feat).max(dim=-1)[0]            # (B, 64, N)

        local = torch.cat((x1, x2, x3), dim=1)          # (B, 192, N)
        g_feat = self.conv4(local).max(dim=-1, keepdim=True)[0]  # (B, 1024, 1)

        if self.conv_cat is not None:
            if label is None:
                raise ValueError("label (B, num_categories) is required when "
                                 "num_categories > 0")
            cat = self.conv_cat(label.view(B, -1, 1))   # (B, 64, 1)
            g_feat = torch.cat((g_feat, cat), dim=1)    # (B, 1088, 1)

        g_feat = g_feat.repeat(1, 1, N)
        feat = torch.cat((g_feat, x1, x2, x3), dim=1)   # (B, 1280, N)

        feat = self.dp1(self.conv5(feat))
        feat = self.dp2(self.conv6(feat))
        feat = self.conv7(feat)
        return self.conv8(feat)                         # (B, p, N)


# ==========================================================================
# factory
# ==========================================================================
def build_model(task, args, num_classes, num_categories=16,
                gate_all_blocks=False, use_transform=True):
    """
    task: "cls" or "seg"
    num_classes: number of object classes (cls) or per-point part/semantic labels (seg)
    num_categories, use_transform: seg only
    """
    task = task.lower()
    if task in ("cls", "classification"):
        return DGCNN(args, output_channels=num_classes,
                     gate_all_blocks=gate_all_blocks)
    if task in ("seg", "segmentation"):
        return DGCNN_Seg(args, num_classes=num_classes,
                         num_categories=num_categories,
                         gate_all_blocks=gate_all_blocks,
                         use_transform=use_transform)
    raise ValueError(f"unknown task '{task}' (expected 'cls' or 'seg')")


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

    cls = build_model("cls", args, num_classes=40)
    print("cls plain:", tuple(cls(x).shape))
    print("cls gated:", tuple(cls(x, complexity=g).shape), "lambda =", cls.last_lambda)

    seg = build_model("seg", args, num_classes=50, num_categories=16)
    print("seg plain:", tuple(seg(x, label).shape))
    print("seg gated:", tuple(seg(x, label, complexity=g).shape), "lambda =", seg.last_lambda)
