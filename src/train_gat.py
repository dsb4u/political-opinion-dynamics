import torch
import pandas as pd
import networkx as nx
from torch_geometric.utils import from_networkx
from torch_geometric.nn import GATConv
import torch.nn.functional as F
import torch.nn as nn

print("Loading dataset...")
df = pd.read_csv("data/clean_comments_2019_09.csv")

print("Loading embeddings...")
embeddings = torch.load("data/comment_embeddings.pt").float()

print("Embedding shape:", embeddings.shape)

# remove t1_ prefix
df["parent_id"] = df["parent_id"].str.replace("t1_", "", regex=False)

print("Building reply graph...")

print("Creating node index mapping...")

id_to_index = {cid: i for i, cid in enumerate(df["id"])}

print("Total mapped nodes:", len(id_to_index))

print("Building reply graph...")

edges = []

for _, row in df.iterrows():

    parent = row["parent_id"]
    child = row["id"]

    if parent in id_to_index:

        edges.append([
            id_to_index[parent],
            id_to_index[child]
        ])

edge_index = torch.tensor(edges).t().contiguous()

print("Edges:", edge_index.shape[1])



print("Converting graph to PyTorch format...")
from torch_geometric.data import Data

data = Data(
    x=embeddings,
    edge_index=edge_index
)

data.x = embeddings

print(data)


# GAT MODEL


class OpinionGAT(nn.Module):

    def __init__(self):

        super().__init__()

        
        self.project = nn.Linear(768, 128)

        
        self.gat1 = GATConv(128, 128, heads=4)
        self.gat2 = GATConv(512, 128)

    def forward(self, x, edge_index):

        # project embeddings
        x = self.project(x)

        
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = self.gat2(x, edge_index)

        return x

# end

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = OpinionGAT().to(device)

data = data.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("Starting training...")

for epoch in range(20):

    optimizer.zero_grad()

    out = model(data.x, data.edge_index)

    target = model.project(data.x)

    loss = F.mse_loss(out, target)

    loss.backward()

    optimizer.step()

    print(f"Epoch {epoch} | Loss {loss.item():.4f}")