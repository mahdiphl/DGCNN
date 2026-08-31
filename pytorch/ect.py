from local_ect import compute_local_ect
import torch
from torch_geometric.datasets import ModelNet
from torch_geometric.transforms import Compose, SamplePoints, KNNGraph, NormalizeScale, FaceToEdge
from visualization import visualize_ect


class PosToX:
    def __call__(self, data):
        data.x = data.pos
        return data

transform = Compose([
    SamplePoints(num=10000),
    KNNGraph(k=6),
    #NormalizeScale(),
    PosToX(),
])

dataset = ModelNet(root='data/ModelNet10', name='10', train=True, transform=transform)


ect = compute_local_ect(dataset, ECT_TYPE = 'edges')   
visualize_ect(ect, save_path='./Figs')
print(ect.shape)