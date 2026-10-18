
"""
DGCNN with BOTH ECT-based mechanisms active at the same time:

  (A) Edge reweighting   (edge_weights.py)
        inside every gated EdgeConv, each edge (i, j) output is scaled by
            w_ij = 1 + lambda * (g_j - g_i)        (or the symmetric MLP variant)
        BEFORE the max over neighbours.  g is the per-point ECT complexity.

  (B) Nested block gating (nested_gating.py)
        after the EdgeConv block produced u, an extra same-width EdgeConv gives v,
        and a per-point gate g_i = sigmoid(alpha * (s_i - median_B(s)))  blends
            f = (1 - g) * u + g * v

Per gated block r the combined computation is

    e_ij   = EC_r(f^(r-1))_ij
    e_ij  <- w_ij * e_ij                                   (A) edge reweighting
    u_i    = max_j e_ij
    v_i    = max_j  EC_extra,r(u)_ij  [optionally also edge-reweighted]
    f_i    = (1 - g_i) u_i + g_i v_i                       (B) nested blending

so (A) changes WHICH neighbour wins the max / how strong each edge is, while
(B) decides HOW MUCH extra refinement a point receives.  Both consume the same
precomputed (B, N) ECT score; DGCNN never subsamples, so it can be reused at
every block unchanged.

API (same contract as the two original main files):
    model  = build_model(task, args, num_classes, ...)
    logits = model(x, complexity=c)            # cls
    logits = model(x, label, complexity=c)     # seg
    complexity=None -> plain ungated DGCNN baseline (identical to stock).

Ablation switches (all in one model):
    use_edge_gate=False   -> nested gating only
    use_nested_gate=False -> edge reweighting only
    both True             -> combined model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# kNN / edge features (returns idx so the edge gate reuses the same graph)
# ---------------------------------------------------------------------------
def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    return pairwise_distance.topk(k=k, dim=-1)[1]          # (B, N, k)


def get_graph_feature(x, k=20, idx=None):
    """Returns edge features (B, 2C, N, k) and neighbour idx (B, N, k)."""
    B, _, N = x.size()
    x = x.view(B, -1, N)
    if idx is None:
        idx = knn(x, k=k)
    idx_raw = idx

    idx_base = torch.arange(0, B, device=x.device).view(-1, 1, 1) * N
    idx_off = (idx + idx_base).view(-1)

    C = x.size(1)
    x_t = x.transpose(2, 1).contiguous()
    feature = x_t.view(B * N, -1)[idx_off, :].view(B, N, k, C)
    x_rep = x_t.view(B, N, 1, C).repeat(1, 1, k, 1)
    feature = torch.cat((feature - x_rep, x_rep), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature, idx_raw


def gather_node_gate(g, idx):
    """g: (B, N) per-point score; idx: (B, N, k). Returns g_i, g_j: (B, N, k)."""
    B, N, k = idx.shape
    idx_base = torch.arange(0, B, device=g.device).view(-1, 1, 1) * N
    idx_flat = (idx + idx_base).view(-1)
    g_j = g.reshape(B * N)[idx_flat].view(B, N, k)
    g_i = g.unsqueeze(-1).expand(-1, -1, k)
    return g_i, g_j


# ---------------------------------------------------------------------------
# (A) edge gate
# ---------------------------------------------------------------------------
class EdgeGate(nn.Module):
    """w_ij = 1 + lambda (g_j - g_i)  |  w_ij = sigmoid(MLP([g_i, g_j]))"""

    def __init__(self, mode="asymmetric", lambda_init=1.0, hidden_dim=8):
        super().__init__()
        assert mode in ("asymmetric", "symmetric")
        self.mode = mode
        if mode == "asymmetric":
            self.lam = nn.Parameter(torch.tensor(float(lambda_init)))
        else:
            self.mlp = nn.Sequential(nn.Linear(2, hidden_dim), nn.ReLU(inplace=True),
                                     nn.Linear(hidden_dim, 1))

    def forward(self, g_i, g_j):
        if self.mode == "asymmetric":
            w = 1.0 + self.lam * (g_j - g_i)
        else:
            w = torch.sigmoid(self.mlp(torch.stack([g_i, g_j], dim=-1))).squeeze(-1)
        return w.unsqueeze(1)                                # (B, 1, N, k)


# ---------------------------------------------------------------------------
# (B) nested gate
# ---------------------------------------------------------------------------
class NestedGate(nn.Module):
    """g_i = sigmoid(alpha (s_i - med_B(s)))  or  sigmoid(MLP([f_i; s_i]))."""

    def __init__(self, in_channels, alpha=4.0, learned=False):
        super().__init__()
        self.alpha = alpha
        self.learned = learned
        if learned:
            self.mlp = nn.Sequential(nn.Linear(in_channels + 1, in_channels // 2),
                                     nn.LeakyReLU(negative_slope=0.2),
                                     nn.Linear(in_channels // 2, 1))

    def forward(self, feat, score):
        """feat (B, C, N), score (B, N) -> (B, 1, N)"""
        if self.learned:
            h = torch.cat([feat.transpose(2, 1), score.unsqueeze(-1)], dim=-1)
            return torch.sigmoid(self.mlp(h)).transpose(2, 1)
        med = score.median(dim=1, keepdim=True)[0]
        return torch.sigmoid(self.alpha * (score - med)).unsqueeze(1)


# ---------------------------------------------------------------------------
# shared machinery for cls and seg models
# ---------------------------------------------------------------------------
class _CombinedGatingMixin:
    """
    Expects the subclass to define self.k and self.<name> EdgeConv modules and
    to call self._init_gating(...) with the per-block output widths.
    """

    def _init_gating(self, block_names, block_channels, gate_all_blocks,
                     use_edge_gate, use_nested_gate, edge_all_blocks,
                     nested_all_blocks, gate_mode, lambda_init,
                     nested_alpha, learned_nested_gate, edge_on_extra,
                     track_diagnostics):
        self.block_names = list(block_names)
        self.use_edge_gate = use_edge_gate
        self.use_nested_gate = use_nested_gate
        self.gate_mode = gate_mode
        self.edge_on_extra = edge_on_extra
        self.track_diagnostics = track_diagnostics
        self.diagnostics = {}

        # per-mechanism block selection; falls back to gate_all_blocks
        edge_all = gate_all_blocks if edge_all_blocks is None else edge_all_blocks
        nest_all = gate_all_blocks if nested_all_blocks is None else nested_all_blocks
        first = self.block_names[:1]
        self.edge_blocks = (self.block_names if edge_all else first) if use_edge_gate else []
        self.nested_blocks = (self.block_names if nest_all else first) if use_nested_gate else []

        # edge gates (tiny; created for every block so checkpoints stay
        # compatible if the block selection flag changes)
        self.edge_gates = nn.ModuleDict(
            {b: EdgeGate(gate_mode, lambda_init) for b in self.block_names})

        # nested: one same-width extra EdgeConv + NestedGate per nested block
        self.extra_convs = nn.ModuleDict()
        self.nested_gates = nn.ModuleDict()
        self.extra_edge_gates = nn.ModuleDict()
        for b in self.nested_blocks:
            c = block_channels[b]
            self.extra_convs[b] = nn.Sequential(
                nn.Conv2d(c * 2, c, kernel_size=1, bias=False),
                nn.BatchNorm2d(c), nn.LeakyReLU(negative_slope=0.2))
            self.nested_gates[b] = NestedGate(c, alpha=nested_alpha, learned=learned_nested_gate)
            if edge_on_extra and use_edge_gate:
                self.extra_edge_gates[b] = EdgeGate(gate_mode, lambda_init)

        self.last_gates = {}

    # -- logging helpers ----------------------------------------------------
    @property
    def last_lambda(self):
        """Mean edge-gate lambda over active edge blocks, read from the live
        nn.Parameters (correct under DataParallel). None if not applicable."""
        if self.gate_mode != "asymmetric" or not self.edge_blocks:
            return None
        return sum(self.edge_gates[b].lam.item() for b in self.edge_blocks) / len(self.edge_blocks)

    @property
    def last_gate_mean(self):
        """Mean nested-gate value of the last forward. Cached python state, so
        it is only reliable on a single device (DataParallel replicas drop it)."""
        if not self.last_gates:
            return None
        return torch.cat([g.reshape(-1) for g in self.last_gates.values()]).mean().item()

    def shape_level_gate_score(self, block=None):
        if not self.last_gates:
            raise RuntimeError("Run a forward pass with complexity != None first.")
        if block is not None:
            return self.last_gates[block].mean(dim=1)
        return torch.stack(list(self.last_gates.values()), 0).mean(0).mean(1)

    # -- core block ---------------------------------------------------------
    def _edge_reweight(self, edge_out, gate, g, idx, name):
        g_i, g_j = gather_node_gate(g, idx)
        gated = edge_out * gate(g_i, g_j)
        if self.track_diagnostics:
            with torch.no_grad():
                self.diagnostics[name] = (edge_out.argmax(-1) != gated.argmax(-1)).float().mean().item()
        return gated

    def _block(self, x, conv, name, complexity):
        """EdgeConv (+ edge reweighting) -> max -> (+ nested blending)."""
        feature, idx = get_graph_feature(x, k=self.k)
        edge_out = conv(feature)                                   # (B, C, N, k)

        if complexity is not None and name in self.edge_blocks:
            edge_out = self._edge_reweight(edge_out, self.edge_gates[name],
                                           complexity, idx, name)
        u = edge_out.max(dim=-1, keepdim=False)[0]                 # (B, C, N)

        if complexity is None or name not in self.nested_blocks:
            return u

        # nested branch: v = EC_extra(u), blended with u by the gate
        feat2, idx2 = get_graph_feature(u, k=self.k)
        e2 = self.extra_convs[name](feat2)
        if name in self.extra_edge_gates:
            e2 = self._edge_reweight(e2, self.extra_edge_gates[name],
                                     complexity, idx2, name + "_extra")
        v = e2.max(dim=-1, keepdim=False)[0]

        g = self.nested_gates[name](u, complexity)                 # (B, 1, N)
        self.last_gates[name] = g.squeeze(1)
        return (1 - g) * u + g * v

    def _reset_state(self, complexity):
        if complexity is not None:
            self.last_gates = {}
            if self.track_diagnostics:
                self.diagnostics = {}


_GATING_KW = dict(gate_all_blocks=False, use_edge_gate=True, use_nested_gate=True,
                  edge_all_blocks=None, nested_all_blocks=None,
                  gate_mode="asymmetric", lambda_init=1.0, nested_alpha=4.0,
                  learned_nested_gate=False, edge_on_extra=False,
                  track_diagnostics=False)


# ---------------------------------------------------------------------------
# classification model
# ---------------------------------------------------------------------------
class DGCNN(_CombinedGatingMixin, nn.Module):
    def __init__(self, args, output_channels=40, **kw):
        super().__init__()
        cfg = {**_GATING_KW, **kw}
        self.args = args
        self.k = args.k

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, 1, bias=False), self.bn1, nn.LeakyReLU(0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(128, 64, 1, bias=False), self.bn2, nn.LeakyReLU(0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(128, 128, 1, bias=False), self.bn3, nn.LeakyReLU(0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(256, 256, 1, bias=False), self.bn4, nn.LeakyReLU(0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, 1, bias=False), self.bn5,
                                   nn.LeakyReLU(0.2))

        self._init_gating(['conv1', 'conv2', 'conv3', 'conv4'],
                          {'conv1': 64, 'conv2': 64, 'conv3': 128, 'conv4': 256}, **cfg)

        self.linear1 = nn.Linear(args.emb_dims * 2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x, complexity=None):
        """x (B, 3, N); complexity (B, N) in [0, 1] or None (plain DGCNN)."""
        B = x.size(0)
        self._reset_state(complexity)

        x1 = self._block(x, self.conv1, 'conv1', complexity)
        x2 = self._block(x1, self.conv2, 'conv2', complexity)
        x3 = self._block(x2, self.conv3, 'conv3', complexity)
        x4 = self._block(x3, self.conv4, 'conv4', complexity)

        x = self.conv5(torch.cat((x1, x2, x3, x4), dim=1))
        x = torch.cat((F.adaptive_max_pool1d(x, 1).view(B, -1),
                       F.adaptive_avg_pool1d(x, 1).view(B, -1)), 1)

        x = self.dp1(F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2))
        x = self.dp2(F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2))
        return self.linear3(x)


# ---------------------------------------------------------------------------
# segmentation model
# ---------------------------------------------------------------------------
class TransformNet(nn.Module):
    """Spatial transform (not gated)."""

    def __init__(self, k=20):
        super().__init__()
        self.k = k
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, 1, bias=False), self.bn1, nn.LeakyReLU(0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, 1, bias=False), self.bn2, nn.LeakyReLU(0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, 1, bias=False), self.bn3, nn.LeakyReLU(0.2))
        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn4 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 256, bias=False)
        self.bn5 = nn.BatchNorm1d(256)
        self.transform = nn.Linear(256, 9)
        nn.init.constant_(self.transform.weight, 0)
        nn.init.eye_(self.transform.bias.view(3, 3))

    def forward(self, x):
        B = x.size(0)
        t, _ = get_graph_feature(x, k=self.k)
        t = self.conv2(self.conv1(t)).max(dim=-1, keepdim=False)[0]
        t = self.conv3(t).max(dim=-1, keepdim=False)[0]
        t = F.leaky_relu(self.bn4(self.linear1(t)), negative_slope=0.2)
        t = F.leaky_relu(self.bn5(self.linear2(t)), negative_slope=0.2)
        t = self.transform(t).view(B, 3, 3)
        return torch.bmm(x.transpose(2, 1), t).transpose(2, 1)


class DGCNN_Seg(_CombinedGatingMixin, nn.Module):
    def __init__(self, args, num_classes=50, num_categories=16, use_transform=True, **kw):
        super().__init__()
        cfg = {**_GATING_KW, **kw}
        self.args = args
        self.k = args.k
        self.num_categories = num_categories
        self.transform_net = TransformNet(k=args.k) if use_transform else None

        def cbr2(i, o):
            return [nn.Conv2d(i, o, 1, bias=False), nn.BatchNorm2d(o), nn.LeakyReLU(0.2)]

        self.conv1 = nn.Sequential(*cbr2(6, 64), *cbr2(64, 64))
        self.conv2 = nn.Sequential(*cbr2(128, 64), *cbr2(64, 64))
        self.conv3 = nn.Sequential(*cbr2(128, 64))

        self._init_gating(['conv1', 'conv2', 'conv3'],
                          {'conv1': 64, 'conv2': 64, 'conv3': 64}, **cfg)

        self.conv4 = nn.Sequential(nn.Conv1d(192, args.emb_dims, 1, bias=False),
                                   nn.BatchNorm1d(args.emb_dims), nn.LeakyReLU(0.2))
        if num_categories > 0:
            self.conv_cat = nn.Sequential(nn.Conv1d(num_categories, 64, 1, bias=False),
                                          nn.BatchNorm1d(64), nn.LeakyReLU(0.2))
            head_in = args.emb_dims + 64 + 192
        else:
            self.conv_cat = None
            head_in = args.emb_dims + 192

        self.conv5 = nn.Sequential(nn.Conv1d(head_in, 256, 1, bias=False),
                                   nn.BatchNorm1d(256), nn.LeakyReLU(0.2))
        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv6 = nn.Sequential(nn.Conv1d(256, 256, 1, bias=False),
                                   nn.BatchNorm1d(256), nn.LeakyReLU(0.2))
        self.dp2 = nn.Dropout(p=args.dropout)
        self.conv7 = nn.Sequential(nn.Conv1d(256, 128, 1, bias=False),
                                   nn.BatchNorm1d(128), nn.LeakyReLU(0.2))
        self.conv8 = nn.Conv1d(128, num_classes, 1, bias=True)

    def forward(self, x, label=None, complexity=None):
        B, _, N = x.size()
        self._reset_state(complexity)

        if self.transform_net is not None:
            x = self.transform_net(x)

        x1 = self._block(x, self.conv1, 'conv1', complexity)
        x2 = self._block(x1, self.conv2, 'conv2', complexity)
        x3 = self._block(x2, self.conv3, 'conv3', complexity)

        g_feat = self.conv4(torch.cat((x1, x2, x3), dim=1)).max(dim=-1, keepdim=True)[0]
        if self.conv_cat is not None:
            if label is None:
                raise ValueError("label (B, num_categories) is required when num_categories > 0")
            g_feat = torch.cat((g_feat, self.conv_cat(label.view(B, -1, 1))), dim=1)
        g_feat = g_feat.repeat(1, 1, N)

        feat = torch.cat((g_feat, x1, x2, x3), dim=1)
        feat = self.dp1(self.conv5(feat))
        feat = self.dp2(self.conv6(feat))
        return self.conv8(self.conv7(feat))                         # (B, p, N)


def build_model(task, args, num_classes, num_categories=16, use_transform=True, **gating_kw):
    """task: 'cls' or 'seg'. gating_kw: see _GATING_KW."""
    task = task.lower()
    if task in ("cls", "classification"):
        return DGCNN(args, output_channels=num_classes, **gating_kw)
    if task in ("seg", "segmentation"):
        return DGCNN_Seg(args, num_classes=num_classes, num_categories=num_categories,
                         use_transform=use_transform, **gating_kw)
    raise ValueError("unknown task '%s' (expected 'cls' or 'seg')" % task)


if __name__ == "__main__":
    class Args:
        k = 20
        emb_dims = 1024
        dropout = 0.5

    args = Args()
    B, N = 4, 256
    x = torch.randn(B, 3, N)
    c = torch.rand(B, N)
    label = F.one_hot(torch.randint(0, 16, (B,)), 16).float()

    for edge, nest, allb in [(True, True, False), (True, True, True),
                             (True, False, True), (False, True, True)]:
        kw = dict(use_edge_gate=edge, use_nested_gate=nest, gate_all_blocks=allb)
        cls = build_model("cls", args, 40, **kw)
        seg = build_model("seg", args, 50, **kw)
        print("edge=%s nested=%s all_blocks=%s" % (edge, nest, allb))
        print("  cls plain:", tuple(cls(x).shape))
        print("  cls gated:", tuple(cls(x, complexity=c).shape),
              "lambda =", cls.last_lambda, "gate mean =", cls.last_gate_mean)
        print("  seg gated:", tuple(seg(x, label, complexity=c).shape),
              "lambda =", seg.last_lambda, "gate mean =", seg.last_gate_mean)
