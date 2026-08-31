

import torch
from torch_geometric.data import Batch, Data
from torch_geometric.datasets import Planetoid, HeterophilousGraphDataset, Amazon, Reddit, WebKB, WikipediaNetwork, Actor, LINKXDataset, WikiCS, Coauthor
import numpy as np
from torch_geometric.utils import k_hop_subgraph
from matplotlib import pyplot

from layers.ect import EctLayer
from layers.config import EctConfig

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score
import xgboost as xgb


def compute_local_ect(dataset,
                      radius=1,
                      ECT_TYPE='points',
                      NUM_THETAS = 64,
                      DEVICE = 'cpu',
                      subsample_size=None
):
 

    data = dataset
    #data = dataset[1]
    features = data.x
    if subsample_size != None:
        np.random.seed(42)
        idx = np.random.choice(
                     range(len(data.x)),
                     replace=False,
                     size=subsample_size,
                 )

        sub_nodes = np.array(range(len(data.x)))[idx]
    else:
        sub_nodes = np.array(range(len(data.x)))

    batch = Batch.from_data_list(
        [
            Data(x=(data.x)[k_hop_subgraph(int(i), radius, data.edge_index, relabel_nodes=True)[0]],
                 edge_index=k_hop_subgraph(int(i), radius, data.edge_index, relabel_nodes=True)[1])
            for i in list(sub_nodes)
        ]
    ).to(DEVICE)

    CONFIG = EctConfig(num_thetas=NUM_THETAS, bump_steps=NUM_THETAS,
                       normalized=False, device=DEVICE, num_features=features.shape[1], ect_type=ECT_TYPE)

    ectlayer = EctLayer(config=CONFIG)

    ect = ectlayer(batch)
    ect = ect.reshape(ect.shape[0], ect.shape[1] * ect.shape[2])
    
    return ect



#################################################################################
