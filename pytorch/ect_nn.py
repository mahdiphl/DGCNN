import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data
from torch_geometric.datasets import HeterophilousGraphDataset, WebKB, Planetoid
from torch_geometric.utils.subgraph import k_hop_subgraph
from layers.ect import EctLayer
from layers.config import EctConfig

# Configuration parameters
radius = 1
ECT_TYPE = 'points'
NUM_THETAS = 64
DEVICE = 'cpu'

# Load dataset
dataset = WebKB(root='/tmp/Wisconsin', name='Wisconsin')
data = dataset[0]
features = data.x

CONFIG = EctConfig(num_thetas=NUM_THETAS, bump_steps=NUM_THETAS,
                   normalized=True, device=DEVICE, num_features=features.shape[1], ect_type=ECT_TYPE)

if len(data.train_mask.shape) > 1:
    train_nodes = np.array(range(len(data.x)))[data.train_mask[:, 0]]
    test_nodes = np.array(range(len(data.x)))[data.test_mask[:, 0]]
    val_nodes = np.array(range(len(data.x)))[data.val_mask[:, 0]]

    labels_train = data.y[data.train_mask[:, 0]]
    labels_test = data.y[data.test_mask[:, 0]]
    labels_val = data.y[data.val_mask[:, 0]]
else:
    train_nodes = np.array(range(len(data.x)))[data.train_mask]
    test_nodes = np.array(range(len(data.x)))[data.test_mask]
    val_nodes = np.array(range(len(data.x)))[data.val_mask]

    labels_train = data.y[data.train_mask]
    labels_test = data.y[data.test_mask]
    labels_val = data.y[data.val_mask]

batch_train = Batch.from_data_list(
    [
        Data(x=(data.x)[k_hop_subgraph(int(i), radius, data.edge_index, relabel_nodes=True)[0]],
             edge_index=k_hop_subgraph(int(i), radius, data.edge_index, relabel_nodes=True)[1])
        for i in list(train_nodes)
    ]
).to(DEVICE)

batch_test = Batch.from_data_list(
    [
        Data(x=(data.x)[k_hop_subgraph(int(i), radius, data.edge_index, relabel_nodes=True)[0]],
             edge_index=k_hop_subgraph(int(i), radius, data.edge_index, relabel_nodes=True)[1])
        for i in list(test_nodes)
    ]
).to(DEVICE)

batch_val = Batch.from_data_list(
    [
        Data(x=(data.x)[k_hop_subgraph(int(i), radius, data.edge_index, relabel_nodes=True)[0]],
             edge_index=k_hop_subgraph(int(i), radius, data.edge_index, relabel_nodes=True)[1])
        for i in list(val_nodes)
    ]
).to(DEVICE)

features_train = torch.tensor([list(features[i]) for i in list(train_nodes)])
features_test = torch.tensor([list(features[i]) for i in list(test_nodes)])
features_val = torch.tensor([list(features[i]) for i in list(val_nodes)])


class SubgraphDataset(Dataset):
    def __init__(self, data, nodes, features, radius):
        self.data = data
        self.nodes = nodes
        self.features = features
        self.radius = radius

    def __len__(self):
        return len(self.nodes)

    def __getitem__(self, idx):
        node = self.nodes[idx]
        sub_nodes, sub_edge_index, _, _ = k_hop_subgraph(int(node), self.radius, self.data.edge_index, relabel_nodes=True)
        sub_data = Data(x=self.data.x[sub_nodes], edge_index=sub_edge_index)
        feature = self.features[idx]
        label = self.data.y[node].item()  # Ensure it's a scalar
        return sub_data, feature, label

radius = 1
train_dataset = SubgraphDataset(data, train_nodes, features_train, radius)
val_dataset = SubgraphDataset(data, val_nodes, features_val, radius)
test_dataset = SubgraphDataset(data, test_nodes, features_test, radius)

def custom_collate(batch):
    subgraphs, features, labels = zip(*batch)
    batched_subgraphs = Batch.from_data_list(subgraphs)
    features = torch.stack(features)  # Stack features into a tensor
    labels = torch.tensor(labels, dtype=torch.long)  # Ensure labels are long tensors
    return batched_subgraphs, features, labels

class ECTModel(nn.Module):
    def __init__(self, hidden_dim, num_classes, feature_dim, dropout_rate=0.5):
        super(ECTModel, self).__init__()
        self.ect_layer = EctLayer(config=CONFIG)
        self.mlp1 = nn.Linear(CONFIG.num_thetas * CONFIG.num_thetas + feature_dim, hidden_dim)
        self.mlp2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.output_layer = nn.Linear(hidden_dim, num_classes)
        self.skip_connection1 = nn.Linear(CONFIG.num_thetas * CONFIG.num_thetas + feature_dim, hidden_dim)
        self.skip_connection2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, batch, features):
        x = self.ect_layer(batch)
        x = x.reshape(x.size(0), -1)  # Flatten the tensor
        x = torch.cat((x, features), dim=1)  # Concatenate the additional features

        # First MLP layer with skip connection
        identity = self.skip_connection1(x)
        x = F.relu(self.mlp1(x))
        x = self.layer_norm1(x + identity)  # Add skip connection
        x = self.dropout(x)

        # Second MLP layer with skip connection
        identity = self.skip_connection2(x)
        x = F.relu(self.mlp2(x))
        x = self.layer_norm2(x + identity)  # Add skip connection
        x = self.dropout(x)

        x = self.output_layer(x)
        return x

def evaluate_model(model, dataloader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():  # Disable gradient computation for evaluation
        for batch, features, labels in dataloader:
            batch = batch.to(device)
            features = features.to(device)
            labels = labels.to(device)
            outputs = model(batch, features)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * batch.num_graphs

            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = running_loss / total_samples
    accuracy = correct_predictions / total_samples
    return avg_loss, accuracy

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=100):
    model.to(device)
    best_val_accuracy = 0.0

    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0

        for batch, features, labels in train_loader:
            batch = batch.to(device)
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch, features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch.num_graphs

        train_loss = running_loss / len(train_loader.dataset)
        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_model.pth')

    print(f"Training complete. Best Val Accuracy: {best_val_accuracy:.4f}")

def test_model(model, test_loader, criterion, device):
    model.load_state_dict(torch.load('best_model.pth'))
    test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)

hidden_dim = 256  # Example hidden layer dimension
num_classes = dataset.num_classes  # Example number of classes
feature_dim = features.shape[1]  # Dimension of additional features
dropout_rate = 0.5  # Dropout rate
learning_rate = 0.001  # Learning rate
num_epochs = 200  # Number of epochs

model = ECTModel(hidden_dim=hidden_dim, num_classes=num_classes, feature_dim=feature_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs)
test_model(model, test_loader, criterion, device)
