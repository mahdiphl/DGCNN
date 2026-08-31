import torch
import torch_geometric
from torch_geometric.data import Batch, Data
import torch.nn as nn
import torch.optim as optim
import geotorch

from layers.ect import EctLayer
from layers.config import EctConfig
from sklearn.neighbors import KDTree

dim = 768
in_size = dim
out_size = dim
DEVICE = 'cpu'


def compute_ect(X,NUM_THETAS=64):
    CONFIG = EctConfig(num_thetas=NUM_THETAS, bump_steps=NUM_THETAS, normalized=True, device=DEVICE, num_features=dim)
    ectlayer = EctLayer(config=CONFIG)

    X = X.requires_grad_(True)

    batch = Batch.from_data_list(
        [
            Data(x=X)
        ]
    ).to(DEVICE)

    res = ectlayer(batch)
    return res[0]


class OrthogonalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(in_size, out_size)
        with torch.no_grad():
            self.linear.weight = nn.Parameter(torch.tensor(np.identity(dim)).type(torch.float32))
        geotorch.sphere(self.linear, "weight")
        geotorch.orthogonal(self.linear, "weight")

    def forward(self, x):
        x = self.linear(x)
        x = compute_ect(x)
        return x


def rotation_inv_distance(X, X_rot):
    X = torch.tensor(X)
    X = X.type(torch.float32)
    X = X.requires_grad_(True)
    X_rot = torch.tensor(X_rot)
    X_rot = X_rot.type(torch.float32)
    X_rot = X_rot.requires_grad_(True)

    # Create an instance of the model
    model = OrthogonalModel()
    # Forward pass
    output = model(X_rot)

    # Print the output
    # print("Output:")
    # print(output)

    # Sample usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define your optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.05)

    num_epochs = 50

    # Sample training loop
    for epoch in range(num_epochs):
        model.train()

        optimizer.zero_grad()

        outputs = model(X_rot)

        loss = torch.square(torch.abs(torch.sub(compute_ect(X), outputs))).sum()
        # loss = torch.abs(torch.sub(X,outputs)).mean()

        loss.backward()
        optimizer.step()

        # print(loss)
        # print(model.linear.weight.T)
    X_rec = torch.matmul(X_rot, model.linear.weight.T)
    X_rec = X_rec.detach().numpy()
    return (torch.square(torch.abs(torch.sub(compute_ect(X), outputs))).sum(), X_rec, model.linear.weight.T)
