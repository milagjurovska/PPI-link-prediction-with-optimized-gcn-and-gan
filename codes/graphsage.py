from sklearn.metrics import f1_score, roc_auc_score, ndcg_score
import torch
from torch_geometric.nn import SAGEConv
import torch.optim as optim
import numpy as np
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def decode(z, edge_label_index):
    src = z[edge_label_index[0]]
    dst = z[edge_label_index[1]]
    return (src * dst).sum(dim=-1)


class GraphSAGELinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=3, dropout=0.5):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.layers.append(SAGEConv(in_channels, hidden_channels, normalize=True))
        for _ in range(num_layers - 2):
            self.layers.append(SAGEConv(hidden_channels, hidden_channels, normalize=True))
        self.layers.append(SAGEConv(hidden_channels, hidden_channels, normalize=True))
        self.dropout = dropout

    def encode(self, x, edge_index):
        for layer in self.layers[:-1]:
            x = layer(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.layers[-1](x, edge_index)


model = GraphSAGELinkPredictor(in_channels=5, hidden_channels=256, num_layers=3).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)


def GraphSAGEtrain(model, optimizer, train_data):
    model.train()
    optimizer.zero_grad()

    x = train_data.x.to(device)
    edge_index = train_data.edge_index.to(device)
    edge_label_index = train_data.edge_label_index.to(device)
    edge_label = train_data.edge_label.to(device)

    z = model.encode(x, edge_index)
    pos_mask = edge_label == 1

    pos_out = decode(z, edge_label_index[:, pos_mask])
    neg_out = decode(z, edge_label_index[:, ~pos_mask])

    eps = 1e-15
    pos_loss = -torch.log(torch.sigmoid(pos_out) + eps).mean()
    neg_loss = -torch.log(1 - torch.sigmoid(neg_out) + eps).mean()
    total_loss = pos_loss + neg_loss

    total_loss.backward()
    optimizer.step()

    return total_loss.item()


@torch.no_grad()
def GraphSAGEtest(model, data):
    model.eval()
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    edge_label_index = data.edge_label_index.to(device)

    z = model.encode(x, edge_index)
    scores = decode(z, edge_label_index)
    probs = torch.sigmoid(scores)
    return probs.cpu().numpy()


def find_optimal_threshold(labels, probs):
    if torch.is_tensor(probs):
        probs = probs.cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()

    thresholds = np.linspace(0, 1, 100)
    best_f1 = 0
    best_thresh = 0.5
    for thresh in thresholds:
        preds = (probs > thresh).astype(int)
        f1 = f1_score(labels, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    return best_thresh


def evaluate_model(test_probs, test_labels):
    if torch.is_tensor(test_probs):
        test_probs = test_probs.cpu().numpy()
    if torch.is_tensor(test_labels):
        test_labels = test_labels.cpu().numpy()

    auc = roc_auc_score(test_labels, test_probs)

    optimal_threshold = find_optimal_threshold(test_labels, test_probs)
    preds = (test_probs > optimal_threshold).astype(int)
    f1 = f1_score(test_labels, preds)

    ndcg = ndcg_score([test_labels], [test_probs])

    return auc, f1, ndcg
