
import importlib
import local_ect
importlib.reload(local_ect)
from local_ect import compute_local_ect
from torch_geometric.transforms import (
    Compose,
    SamplePoints,
    KNNGraph,
    NormalizeScale,
)
from torch_geometric.data import Data

class PosToX:
    def __call__(self, data):
        data.x = data.pos
        del(data.pos)
        return data

ect_transform = Compose([
    KNNGraph(k=5),
    NormalizeScale(),
    PosToX(),
])


from visualize_ect_complexity import (
    to_numpy, reshape_ect, compute_complexity,
    fig_complexity_3d, fig_complexity_multiview,
    supp_ect_heatmaps, supp_complexity_histogram,
    make_dirs, MANUSCRIPT_DIR, SUPP_DIR
)


def compute_complexity_score(data, method='entropy'):
    #data = Data(pos=xyz)
    data = ect_transform(data)
    ect = compute_local_ect(data, radius=1, ECT_TYPE='points')
    complexity = compute_complexity(ect, num_thetas=64, method="entropy")
    return complexity

