
#!/usr/bin/env python
"""
DGCNN backbone with ECT-complexity-gated edge reweighting.

API is matched to main.py's expectations:
    model = DGCNN(args, output_channels=40, gate_all_blocks=args.gate_all_blocks)
    logits = model(data, complexity=complexity)   # complexity: (B, N) tensor or None
    model.last_lambda                              # float or None, for logging

complexity=None reproduces the plain, ungated DGCNN forward pass exactly
(no weighting is applied anywhere) -- this is what lets --use_gate False
give you the true baseline for comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# k-NN / edge-feature construction (device-safe version of model.py's helpers)
# ---------------------------------------------------------------------------

def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (B, N, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    """
    Builds DGCNN's edge features. Always returns idx (B, N, k) as well
    (un-offset by batch), so gated blocks can reuse it to gather g_i, g_j
    from the *same* neighborhood that produced these edge features -- no
    second k-NN call, no chance of misalignment between edge features and gate.
    """
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (B, N, k)
    idx_raw = idx

    device = x.device  # fixed vs. original model.py: no hardcoded 'cuda'
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx_off = (idx + idx_base).view(-1)

    _, num_dims, _ = x.size()

    x_t = x.transpose(2, 1).contiguous()
    feature = x_t.view(batch_size * num_points, -1)[idx_off, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x_rep = x_t.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x_rep, x_rep), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature, idx_raw  # (B, 2*num_dims, N, k), (B, N, k)


def gather_node_gate(g, idx):
    """
    g:   (B, N)     per-node gate (ECT complexity score), aligned to the
                    original point ordering.
    idx: (B, N, k)  neighbor indices in [0, N), from get_graph_feature.

    Returns g_i, g_j, each (B, N, k): the gate value of the center node and
    of each of its k neighbors, for every edge.
    """
    batch_size, num_points, k = idx.shape
    device = g.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx_flat = (idx + idx_base).view(-1)

    g_flat = g.reshape(batch_size * num_points)
    g_j = g_flat[idx_flat].view(batch_size, num_points, k)
    g_i = g.unsqueeze(-1).expand(-1, -1, k)
    return g_i, g_j


# ---------------------------------------------------------------------------
# Edge gate
# ---------------------------------------------------------------------------

class EdgeGate(nn.Module):
    """
    w_ij = 1 + lambda * (g_j - g_i)     (mode='asymmetric', lambda learnable)
    w_ij = sigmoid(MLP([g_i, g_j]))     (mode='symmetric', not used by main.py
                                          yet -- kept available if you extend
                                          the argparse with --gate_mode later)
    """

    def __init__(self, mode="asymmetric", lambda_init=1.0, hidden_dim=8):
        super().__init__()
        assert mode in ("asymmetric", "symmetric")
        self.mode = mode
        if mode == "asymmetric":
            self.lam = nn.Parameter(torch.tensor(float(lambda_init)))
        else:
            self.mlp = nn.Sequential(
                nn.Linear(2, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 1),
            )

    def forward(self, g_i, g_j):
        if self.mode == "asymmetric":
            w = 1.0 + self.lam * (g_j - g_i)
        else:
            feat = torch.stack([g_i, g_j], dim=-1)
            w = torch.sigmoid(self.mlp(feat)).squeeze(-1)
        return w.unsqueeze(1)  # (B, 1, N, k), broadcasts over channels


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class DGCNN(nn.Module):
    def __init__(self, args, output_channels=40, gate_all_blocks=False,
                 gate_mode="asymmetric", lambda_init=1.0, track_diagnostics=False):
        super().__init__()
        self.args = args
        self.k = args.k
        self.gate_all_blocks = gate_all_blocks
        self.gate_mode = gate_mode
        self.track_diagnostics = track_diagnostics
        self.diagnostics = {}

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

        # Gate 1 always exists (used whenever gating is on, even gate_all_blocks=False).
        # Gates 2-4 only matter when gate_all_blocks=True, but we create them
        # unconditionally so state_dicts stay compatible if you flip the flag
        # later when resuming from a checkpoint.
        self.gate1 = EdgeGate(mode=gate_mode, lambda_init=lambda_init)
        self.gate2 = EdgeGate(mode=gate_mode, lambda_init=lambda_init)
        self.gate3 = EdgeGate(mode=gate_mode, lambda_init=lambda_init)
        self.gate4 = EdgeGate(mode=gate_mode, lambda_init=lambda_init)

        self.linear1 = nn.Linear(args.emb_dims * 2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    @property
    def last_lambda(self):
        """
        Read directly off the live nn.Parameter rather than caching a value
        during forward(). Under nn.DataParallel with >1 GPU, forward() runs
        on temporary replica modules, and python-attribute writes made
        inside forward() do NOT propagate back to model.module -- only
        parameter/buffer storage is shared. Reading the Parameter itself
        here sidesteps that entirely and is correct regardless of device count.
        """
        if self.gate_mode != "asymmetric":
            return None
        gate = self.gate4 if self.gate_all_blocks else self.gate1
        return gate.lam.item()

    def _edge_conv_block(self, x, conv, gate, g, layer_name):
        feature, idx = get_graph_feature(x, k=self.k)  # (B, 2C, N, k), (B, N, k)
        edge_out = conv(feature)                       # (B, C_out, N, k)

        if g is None:
            return edge_out.max(dim=-1, keepdim=False)[0]

        g_i, g_j = gather_node_gate(g, idx)
        w = gate(g_i, g_j)
        gated_out = edge_out * w

        if self.track_diagnostics:
            with torch.no_grad():
                base_winner = edge_out.argmax(dim=-1)
                gated_winner = gated_out.argmax(dim=-1)
                self.diagnostics[layer_name] = (base_winner != gated_winner).float().mean().item()

        return gated_out.max(dim=-1, keepdim=False)[0]

    def forward(self, x, complexity=None):
        """
        x: (B, 3, N) point coordinates, channel-first.
        complexity: (B, N) per-node ECT complexity score, or None to run the
                    plain, ungated DGCNN baseline (no weighting applied anywhere).
        """
        batch_size = x.size(0)
        use_gate = complexity is not None
        if self.track_diagnostics and use_gate:
            self.diagnostics = {}

        g1 = complexity if use_gate else None
        x1 = self._edge_conv_block(x, self.conv1, self.gate1, g1, "layer1")

        g_rest = complexity if (use_gate and self.gate_all_blocks) else None
        x2 = self._edge_conv_block(x1, self.conv2, self.gate2, g_rest, "layer2")
        x3 = self._edge_conv_block(x2, self.conv3, self.gate3, g_rest, "layer3")
        x4 = self._edge_conv_block(x3, self.conv4, self.gate4, g_rest, "layer4")

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv5(x)
        p1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        p2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((p1, p2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x


# ===========================================================================
# ADDED: segmentation model (bottom branch of the DGCNN paper's Fig. 3) with
# the same ECT-gated edge reweighting as the classification DGCNN above.
# Everything above this line is the original edge_weights.py, unchanged.
# ===========================================================================

class TransformNet(nn.Module):
    """Spatial transform block: estimates a 3x3 matrix from the edge features
    (coords + neighbour differences) and applies it to the input points.
    Not gated."""

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
        t, _ = get_graph_feature(x, k=self.k)    # this file's version also returns idx
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
    DGCNN part-segmentation network (Fig. 3, bottom branch) with ECT-gated
    edge reweighting.

        point cloud (B,3,N) -> spatial transform
          -> EdgeConv mlp{64,64}  -> x1   (64)   [edge gate 1]
          -> EdgeConv mlp{64,64}  -> x2   (64)   [edge gate 2]
          -> EdgeConv mlp{64}     -> x3   (64)   [edge gate 3]
          -> concat(x1,x2,x3) (192) -> mlp{1024} -> max pool     (1024)
          -> concat mlp{64}(category one-hot)                    (1088)
          -> repeat N, concat (x1,x2,x3)                         (1280)
          -> mlp{256,256,128,p}                                  (B,p,N)

    Gating is identical to the classification DGCNN: the full EdgeConv mlp
    output (B, C, N, k) is multiplied by w_ij = 1 + lambda * (g_j - g_i)
    (or the symmetric MLP variant) before the max over neighbours.

    gate_all_blocks=False (default): gate only EdgeConv 1.
    gate_all_blocks=True:            gate all three EdgeConv blocks.

    forward(x, label=None, complexity=None):
        x (B,3,N); label (B,num_categories) one-hot; complexity (B,N) or None
        for the plain ungated baseline.  Returns (B, num_classes, N).

    last_lambda is a property reading the live nn.Parameter (correct under
    multi-GPU DataParallel), exactly as in the classification model.
    """

    def __init__(self, args, num_classes=50, num_categories=16,
                 gate_all_blocks=False, use_transform=True,
                 gate_mode="asymmetric", lambda_init=1.0, track_diagnostics=False):
        super().__init__()
        self.args = args
        self.k = args.k
        self.num_categories = num_categories
        self.gate_all_blocks = gate_all_blocks
        self.gate_mode = gate_mode
        self.track_diagnostics = track_diagnostics
        self.diagnostics = {}

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

        # one edge gate per EdgeConv block (gates 2-3 unused unless gate_all_blocks)
        self.gate1 = EdgeGate(mode=gate_mode, lambda_init=lambda_init)
        self.gate2 = EdgeGate(mode=gate_mode, lambda_init=lambda_init)
        self.gate3 = EdgeGate(mode=gate_mode, lambda_init=lambda_init)

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

    @property
    def last_lambda(self):
        """Live nn.Parameter read (see DGCNN.last_lambda for why)."""
        if self.gate_mode != "asymmetric":
            return None
        gate = self.gate3 if self.gate_all_blocks else self.gate1
        return gate.lam.item()

    def _edge_conv_block(self, x, conv, gate, g, layer_name):
        feature, idx = get_graph_feature(x, k=self.k)  # (B, 2C, N, k), (B, N, k)
        edge_out = conv(feature)                       # (B, C_out, N, k)

        if g is None:
            return edge_out.max(dim=-1, keepdim=False)[0]

        g_i, g_j = gather_node_gate(g, idx)
        w = gate(g_i, g_j)
        gated_out = edge_out * w

        if self.track_diagnostics:
            with torch.no_grad():
                base_winner = edge_out.argmax(dim=-1)
                gated_winner = gated_out.argmax(dim=-1)
                self.diagnostics[layer_name] = (base_winner != gated_winner).float().mean().item()

        return gated_out.max(dim=-1, keepdim=False)[0]

    def forward(self, x, label=None, complexity=None):
        B, _, N = x.size()
        use_gate = complexity is not None
        if self.track_diagnostics and use_gate:
            self.diagnostics = {}

        if self.transform_net is not None:
            x = self.transform_net(x)

        g1 = complexity if use_gate else None
        x1 = self._edge_conv_block(x, self.conv1, self.gate1, g1, "layer1")      # (B, 64, N)

        g_rest = complexity if (use_gate and self.gate_all_blocks) else None
        x2 = self._edge_conv_block(x1, self.conv2, self.gate2, g_rest, "layer2")  # (B, 64, N)
        x3 = self._edge_conv_block(x2, self.conv3, self.gate3, g_rest, "layer3")  # (B, 64, N)

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
        return self.conv8(feat)                                      # (B, p, N)


def build_model(task, args, num_classes, num_categories=16,
                gate_all_blocks=False, use_transform=True,
                gate_mode="asymmetric", lambda_init=1.0, track_diagnostics=False):
    """task: "cls" or "seg".  use_transform / num_categories: seg only."""
    task = task.lower()
    if task in ("cls", "classification"):
        return DGCNN(args, output_channels=num_classes,
                     gate_all_blocks=gate_all_blocks, gate_mode=gate_mode,
                     lambda_init=lambda_init, track_diagnostics=track_diagnostics)
    if task in ("seg", "segmentation"):
        return DGCNN_Seg(args, num_classes=num_classes,
                         num_categories=num_categories,
                         gate_all_blocks=gate_all_blocks,
                         use_transform=use_transform, gate_mode=gate_mode,
                         lambda_init=lambda_init, track_diagnostics=track_diagnostics)
    raise ValueError("unknown task '%s' (expected 'cls' or 'seg')" % task)


if __name__ == "__main__":
    class Args:
        k = 7
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
        print("  cls gated:", tuple(cls(x, complexity=g).shape), "lambda =", cls.last_lambda)
        print("  seg plain:", tuple(seg(x, label).shape))
        print("  seg gated:", tuple(seg(x, label, complexity=g).shape), "lambda =", seg.last_lambda)
