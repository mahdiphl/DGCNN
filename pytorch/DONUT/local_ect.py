
import numpy as np
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.utils import k_hop_subgraph

from layers.ect import EctLayer
from layers.config import EctConfig


def make_directions(num_features=3, num_thetas=64, seed=0):
    """Fixed, uniformly distributed unit directions, shape (num_features, num_thetas).
    Pass the same v to every shape so ECT scores are comparable and reproducible."""
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(num_thetas, num_features, generator=g)
    v = v / v.norm(dim=1, keepdim=True)
    return v.T


def compute_local_ect(dataset,
                      radius=1,
                      ECT_TYPE='points',
                      NUM_THETAS=64,
                      DEVICE='cpu',
                      subsample_size=None,
                      v=None,
                      node_chunk=None):
    """
    v          : optional (num_features, NUM_THETAS) direction tensor. If None,
                 EctLayer draws random directions (unseeded, different per call).
    node_chunk : if set, process this many nodes at a time (bounds memory for
                 large clouds).
    """
    data = dataset
    features = data.x

    if subsample_size is not None:
        np.random.seed(42)
        sub_nodes = np.random.choice(len(data.x), replace=False, size=subsample_size)
    else:
        sub_nodes = np.arange(len(data.x))

    CONFIG = EctConfig(num_thetas=NUM_THETAS, bump_steps=NUM_THETAS,
                       normalized=False, device=DEVICE,
                       num_features=features.shape[1], ect_type=ECT_TYPE)
    if v is not None:
        v = v.to(DEVICE)
    ectlayer = EctLayer(config=CONFIG, v=v)

    def local_graph(i):
        subset, ei, _, _ = k_hop_subgraph(int(i), radius, data.edge_index,
                                          relabel_nodes=True)
        return Data(x=data.x[subset], edge_index=ei)

    if node_chunk is None:
        chunks = [sub_nodes]
    else:
        chunks = np.array_split(sub_nodes, max(1, int(np.ceil(len(sub_nodes) / node_chunk))))

    outs = []
    for ch in chunks:
        batch = Batch.from_data_list([local_graph(i) for i in ch]).to(DEVICE)
        ect = ectlayer(batch)                      # (B, bump_steps, num_thetas)
        outs.append(ect.reshape(ect.shape[0], ect.shape[1] * ect.shape[2]))
    return torch.cat(outs, dim=0)
