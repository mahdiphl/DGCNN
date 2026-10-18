

"""
DGCNN + ECT complexity, NODE gate and EDGE gate used simultaneously.

    from dgcnn_ect_combined import build_model
    model = build_model("cls", args, num_classes=40, gate_type="both")
    model = build_model("seg", args, num_classes=50, num_categories=16, gate_type="both")

Two complementary uses of the per-point ECT complexity score g in [0, 1]:

  NODE gate (feature level, from dgcnn_ect_gated.py)
        f~_i = f_i * (1 + lam_node * g_i)
      rescales the *features of point i* before they enter an EdgeConv
      (raw xyz for block 1, previous block output for later blocks).

  EDGE gate (neighbourhood level, from edge_weights.py)
        w_ij = 1 + lam_edge * (g_j - g_i)
      rescales the *edge MLP output* h(f_i, f_j - f_i) for each of the k edges,
      before the max over neighbours, so complexity changes which neighbour
      "wins" the max-pool.

Order inside one EdgeConv block (gate_type="both"):
    f -> [node gate on f] -> kNN edge features -> MLP -> [edge gate] -> max_j

kNN is computed on the node-gated features (same as the node-gated model);
the edge gate reuses that same idx, so g_i / g_j are always aligned with the
edge features.

gate_type : "none" (plain DGCNN) | "node" | "edge" | "both"
node_gate_all / edge_gate_all : False -> only block 1, True -> every block.
complexity=None always runs the plain baseline, whatever gate_type is.

Each gate has its own lambda (node: init 0.5, edge: init 1.0, as in your
originals). `model.lambdas` -> dict of the live nn.Parameter values (safe under
DataParallel). `model.last_lambda` is kept for backward compatibility.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

GATE_TYPES = ("none", "node", "edge", "both")


# ==========================================================================
# helpers
# ==========================================================================
def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    return pairwise_distance.topk(k=k, dim=-1)[1]  # (B, N, k)


def get_graph_feature(x, k=20, idx=None):
    """Edge features (B, 2C, N, k) and the neighbour idx (B, N, k)."""
    B, N = x.size(0), x.size(2)
    x = x.view(B, -1, N)
    if idx is None:
        idx = knn(x, k=k)
    idx_off = (idx + torch.arange(B, device=x.device).view(-1, 1, 1) * N).view(-1)

    C = x.size(1)
    x_t = x.transpose(2, 1).contiguous()
    nbr = x_t.view(B * N, -1)[idx_off, :].view(B, N, k, C)
    ctr = x_t.view(B, N, 1, C).expand(-1, -1, k, -1)
    feat = torch.cat((nbr - ctr, ctr), dim=3).permute(0, 3, 1, 2).contiguous()
    return feat, idx


def gather_node_gate(g, idx):
    """g (B, N), idx (B, N, k) -> g_i, g_j each (B, N, k)."""
    B, N, k = idx.shape
    idx_flat = (idx + torch.arange(B, device=g.device).view(-1, 1, 1) * N).view(-1)
    g_j = g.reshape(B * N)[idx_flat].view(B, N, k)
    g_i = g.unsqueeze(-1).expand(-1, -1, k)
    return g_i, g_j


# ==========================================================================
# gates
# ==========================================================================
class NodeGate(nn.Module):
    """f~_i = f_i * (1 + lam * g_i).   x: (B, C, N), g: (B, N)."""

    def __init__(self, init_lambda=0.5):
        super().__init__()
        self.lam = nn.Parameter(torch.tensor(float(init_lambda)))

    def forward(self, x, g):
        return x * (1.0 + self.lam * g.unsqueeze(1))


class EdgeGate(nn.Module):
    """asymmetric: w_ij = 1 + lam * (g_j - g_i)
       symmetric : w_ij = sigmoid(MLP([g_i, g_j]))"""

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
        return w.unsqueeze(1)  # (B, 1, N, k)


# ==========================================================================
# shared machinery: one EdgeConv block with optional node + edge gate
# ==========================================================================
class _GatedBackbone(nn.Module):
    """Mixin holding gate bookkeeping shared by the cls and seg models."""

    def _setup_gates(self, n_blocks, gate_type, node_gate_all, edge_gate_all,
                     edge_mode, node_lambda_init, edge_lambda_init, track_diagnostics):
        assert gate_type in GATE_TYPES, f"gate_type must be one of {GATE_TYPES}"
        self.gate_type = gate_type
        self.node_gate_all = node_gate_all
        self.edge_gate_all = edge_gate_all
        self.edge_mode = edge_mode
        self.track_diagnostics = track_diagnostics
        self.diagnostics = {}
        self.n_blocks = n_blocks
        # one gate per block; unused ones keep state_dicts compatible across flags
        self.node_gates = nn.ModuleList([NodeGate(node_lambda_init) for _ in range(n_blocks)])
        self.edge_gates = nn.ModuleList([EdgeGate(edge_mode, edge_lambda_init) for _ in range(n_blocks)])

    def _node_on(self, b):
        return self.gate_type in ("node", "both") and (b == 0 or self.node_gate_all)

    def _edge_on(self, b):
        return self.gate_type in ("edge", "both") and (b == 0 or self.edge_gate_all)

    @property
    def lambdas(self):
        """Live parameter values of the gates that are actually active."""
        out = {}
        if self.gate_type in ("node", "both"):
            last = self.n_blocks - 1 if self.node_gate_all else 0
            out["node"] = self.node_gates[last].lam.item()
        if self.gate_type in ("edge", "both") and self.edge_mode == "asymmetric":
            last = self.n_blocks - 1 if self.edge_gate_all else 0
            out["edge"] = self.edge_gates[last].lam.item()
        return out

    @property
    def last_lambda(self):
        d = self.lambdas
        return d if d else None

    def _block(self, b, x, conv, g):
        """x: (B, C, N) -> (B, C_out, N). g: (B, N) or None (plain)."""
        use = g is not None
        if use and self._node_on(b):
            x = self.node_gates[b](x, g)
        feat, idx = get_graph_feature(x, k=self.k)
        edge_out = conv(feat)                                   # (B, C_out, N, k)
        if use and self._edge_on(b):
            g_i, g_j = gather_node_gate(g, idx)
            gated = edge_out * self.edge_gates[b](g_i, g_j)
            if self.track_diagnostics:
                with torch.no_grad():
                    self.diagnostics[f"layer{b + 1}"] = (
                        edge_out.argmax(-1) != gated.argmax(-1)).float().mean().item()
            edge_out = gated
        return edge_out.max(dim=-1, keepdim=False)[0]


# ==========================================================================
# classification
# ==========================================================================
class DGCNN(_GatedBackbone):
    def __init__(self, args, output_channels=40, gate_type="both",
                 node_gate_all=False, edge_gate_all=False, edge_mode="asymmetric",
                 node_lambda_init=0.5, edge_lambda_init=1.0, track_diagnostics=False):
        super().__init__()
        self.args, self.k = args, args.k
        self._setup_gates(4, gate_type, node_gate_all, edge_gate_all, edge_mode,
                          node_lambda_init, edge_lambda_init, track_diagnostics)

        def blk2d(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 1, bias=False), nn.BatchNorm2d(o),
                                 nn.LeakyReLU(0.2))
        self.conv1, self.conv2 = blk2d(6, 64), blk2d(128, 64)
        self.conv3, self.conv4 = blk2d(128, 128), blk2d(256, 256)
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, 1, bias=False),
                                   nn.BatchNorm1d(args.emb_dims), nn.LeakyReLU(0.2))

        self.linear1 = nn.Linear(args.emb_dims * 2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x, complexity=None):
        """x (B,3,N); complexity (B,N) in [0,1] or None -> (B, output_channels)"""
        B = x.size(0)
        if self.track_diagnostics and complexity is not None:
            self.diagnostics = {}
        x1 = self._block(0, x, self.conv1, complexity)
        x2 = self._block(1, x1, self.conv2, complexity)
        x3 = self._block(2, x2, self.conv3, complexity)
        x4 = self._block(3, x3, self.conv4, complexity)

        f = self.conv5(torch.cat((x1, x2, x3, x4), dim=1))
        f = torch.cat((F.adaptive_max_pool1d(f, 1).view(B, -1),
                       F.adaptive_avg_pool1d(f, 1).view(B, -1)), 1)
        f = self.dp1(F.leaky_relu(self.bn6(self.linear1(f)), 0.2))
        f = self.dp2(F.leaky_relu(self.bn7(self.linear2(f)), 0.2))
        return self.linear3(f)


# ==========================================================================
# segmentation
# ==========================================================================
class TransformNet(nn.Module):
    """Spatial transform (3x3), not gated."""

    def __init__(self, k=20):
        super().__init__()
        self.k = k
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, 1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, 1, bias=False), nn.BatchNorm1d(1024), nn.LeakyReLU(0.2))
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
        t = self.conv2(self.conv1(t)).max(dim=-1)[0]
        t = self.conv3(t).max(dim=-1)[0]
        t = F.leaky_relu(self.bn4(self.linear1(t)), 0.2)
        t = F.leaky_relu(self.bn5(self.linear2(t)), 0.2)
        t = self.transform(t).view(B, 3, 3)
        return torch.bmm(x.transpose(2, 1), t).transpose(2, 1)


class DGCNN_Seg(_GatedBackbone):
    def __init__(self, args, num_classes=50, num_categories=16, gate_type="both",
                 node_gate_all=False, edge_gate_all=False, edge_mode="asymmetric",
                 node_lambda_init=0.5, edge_lambda_init=1.0, use_transform=True,
                 track_diagnostics=False):
        super().__init__()
        self.args, self.k = args, args.k
        self.num_categories = num_categories
        self._setup_gates(3, gate_type, node_gate_all, edge_gate_all, edge_mode,
                          node_lambda_init, edge_lambda_init, track_diagnostics)
        self.transform_net = TransformNet(k=args.k) if use_transform else None

        def c2(i, o):
            return [nn.Conv2d(i, o, 1, bias=False), nn.BatchNorm2d(o), nn.LeakyReLU(0.2)]

        def c1(i, o):
            return [nn.Conv1d(i, o, 1, bias=False), nn.BatchNorm1d(o), nn.LeakyReLU(0.2)]
        self.conv1 = nn.Sequential(*c2(6, 64), *c2(64, 64))
        self.conv2 = nn.Sequential(*c2(128, 64), *c2(64, 64))
        self.conv3 = nn.Sequential(*c2(128, 64))
        self.conv4 = nn.Sequential(*c1(192, args.emb_dims))

        if num_categories > 0:
            self.conv_cat = nn.Sequential(*c1(num_categories, 64))
            head_in = args.emb_dims + 64 + 192
        else:
            self.conv_cat = None
            head_in = args.emb_dims + 192
        self.conv5 = nn.Sequential(*c1(head_in, 256))
        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv6 = nn.Sequential(*c1(256, 256))
        self.dp2 = nn.Dropout(p=args.dropout)
        self.conv7 = nn.Sequential(*c1(256, 128))
        self.conv8 = nn.Conv1d(128, num_classes, 1, bias=True)

    def forward(self, x, label=None, complexity=None):
        """x (B,3,N); label (B,num_categories); complexity (B,N)|None -> (B,p,N)"""
        B, _, N = x.size()
        if self.track_diagnostics and complexity is not None:
            self.diagnostics = {}
        if self.transform_net is not None:
            x = self.transform_net(x)

        x1 = self._block(0, x, self.conv1, complexity)
        x2 = self._block(1, x1, self.conv2, complexity)
        x3 = self._block(2, x2, self.conv3, complexity)

        g_feat = self.conv4(torch.cat((x1, x2, x3), dim=1)).max(dim=-1, keepdim=True)[0]
        if self.conv_cat is not None:
            if label is None:
                raise ValueError("label (B, num_categories) required when num_categories > 0")
            g_feat = torch.cat((g_feat, self.conv_cat(label.view(B, -1, 1))), dim=1)
        feat = torch.cat((g_feat.repeat(1, 1, N), x1, x2, x3), dim=1)

        feat = self.dp1(self.conv5(feat))
        feat = self.dp2(self.conv6(feat))
        return self.conv8(self.conv7(feat))


# ==========================================================================
# factory
# ==========================================================================
def build_model(task, args, num_classes, num_categories=16, gate_type="both",
                node_gate_all=False, edge_gate_all=False, edge_mode="asymmetric",
                node_lambda_init=0.5, edge_lambda_init=1.0, use_transform=True,
                track_diagnostics=False):
    common = dict(gate_type=gate_type, node_gate_all=node_gate_all,
                  edge_gate_all=edge_gate_all, edge_mode=edge_mode,
                  node_lambda_init=node_lambda_init, edge_lambda_init=edge_lambda_init,
                  track_diagnostics=track_diagnostics)
    task = task.lower()
    if task in ("cls", "classification"):
        return DGCNN(args, output_channels=num_classes, **common)
    if task in ("seg", "segmentation"):
        return DGCNN_Seg(args, num_classes=num_classes, num_categories=num_categories,
                         use_transform=use_transform, **common)
    raise ValueError(f"unknown task '{task}' (expected 'cls' or 'seg')")


if __name__ == "__main__":
    class Args:
        k = 20
        emb_dims = 1024
        dropout = 0.5

    args = Args()
    B, N = 4, 256
    x = torch.randn(B, 3, N)
    g = torch.rand(B, N)
    label = F.one_hot(torch.randint(0, 16, (B,)), 16).float()

    for gt in GATE_TYPES:
        for all_b in (False, True):
            c = build_model("cls", args, 40, gate_type=gt, node_gate_all=all_b,
                            edge_gate_all=all_b, track_diagnostics=True).eval()
            s = build_model("seg", args, 50, gate_type=gt, node_gate_all=all_b,
                            edge_gate_all=all_b).eval()
            print(f"{gt:5s} all={all_b!s:5s} cls {tuple(c(x, complexity=g).shape)} "
                  f"seg {tuple(s(x, label, complexity=g).shape)} lambdas={c.lambdas} "
                  f"diag={c.diagnostics}")

