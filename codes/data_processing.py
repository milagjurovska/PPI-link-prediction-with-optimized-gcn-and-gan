import pandas as pd
import networkx as nx
import torch
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import from_networkx
from sklearn.preprocessing import LabelEncoder

import os

# Get the path relative to this script's location
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, "..", "PP-Pathways_ppi.csv", "PP-Pathways_ppi.csv")

df = pd.read_csv(data_path, sep=',', header=None, names=['source', 'target'])

df.dropna(subset=['source', 'target'], inplace=True)

encoder = LabelEncoder()
all_nodes = pd.concat([df['source'], df['target']], axis=0)
encoder.fit(all_nodes)
df['source'] = encoder.transform(df['source'])
df['target'] = encoder.transform(df['target'])

G = nx.from_pandas_edgelist(df, source='source', target='target')
print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

G_data = from_networkx(G)
G_data.num_nodes = G.number_of_nodes()


embed_size = 5
X = torch.nn.Embedding(G.number_of_nodes(), embed_size).weight
X = X.requires_grad_(True)
G_data.x = X

transform = RandomLinkSplit(
    num_val=0.1,
    num_test=0.2,
    is_undirected=True,
    disjoint_train_ratio=0.0,
    neg_sampling_ratio=1.0,
    add_negative_train_samples=True
)
train_data, val_data, test_data = transform(G_data)