

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
