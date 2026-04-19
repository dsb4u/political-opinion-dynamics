import torch
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
import torch.nn.functional as F
import torch.nn as nn

print("Loading dataset...")

df = pd.read_csv("data/clean_comments_2019_09.csv")
# Convert timestamp to datetime
df["created_utc"] = pd.to_datetime(df["created_utc"], unit="s")

# Create 6 hour time windows
window_size = "6h"
df["time_window"] = df["created_utc"].dt.floor(window_size)

# Sort chronologically
df = df.sort_values("created_utc")

print("Time windows created")
print("Number of windows:", df["time_window"].nunique())

print("Loading embeddings...")

embeddings = torch.load("data/comment_embeddings.pt").float()

print("Embedding shape:", embeddings.shape)


# NODE INDEX MAPPING


print("Creating node index mapping...")

id_to_index = {cid: i for i, cid in enumerate(df["id"])}

print("Total nodes:", len(id_to_index))


# BUILD EDGE LIST


print("Building reply graph...")

edges = []

for _, row in df.iterrows():

    parent = row["parent_id"].replace("t1_", "")
    child = row["id"]

    if parent in id_to_index:

        edges.append([
            id_to_index[parent],
            id_to_index[child]
        ])

edge_index = torch.tensor(edges).t().contiguous()

print("Edges:", edge_index.shape[1])

data = Data(
    x=embeddings,
    edge_index=edge_index
)

print(data)


# GLOBAL FEEDBACK g(t)


print("Computing engagement scores...")

df["engagement"] = df["score"] + df["controversiality"]

top_k = 100

top_comments = df.nlargest(top_k, "engagement")

top_indices = [id_to_index[cid] for cid in top_comments["id"]]

g = embeddings[top_indices].mean(dim=0)

print("Global feedback vector shape:", g.shape)


# MODEL


class OpinionDynamicsModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.project = nn.Linear(768, 128)

        self.gat1 = GATConv(128, 128, heads=4)

        self.gat2 = GATConv(512, 128)

        self.decoder = nn.Linear(128, 768)

        self.beta = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, edge_index, g):

        x = self.project(x)

        x = self.gat1(x, edge_index)

        x = F.elu(x)

        x = self.gat2(x, edge_index)

        g = self.project(g)

        x = x + self.beta * g

        x = self.decoder(x)

        return x



# TRAINING


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = OpinionDynamicsModel().to(device)

data = data.to(device)

g = g.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("Starting training...")

windows = sorted(df["time_window"].unique())

for epoch in range(20):

    optimizer.zero_grad()

    pred_all = model(data.x, data.edge_index, g)

    total_loss = 0

    for i in range(len(windows)-1):

        w_t = windows[i]
        w_t1 = windows[i+1]

        idx_t = df[df["time_window"] == w_t].index
        idx_t1 = df[df["time_window"] == w_t1].index

        min_size = min(len(idx_t), len(idx_t1))

        pred = pred_all[idx_t[:min_size]]
        target = data.x[idx_t1[:min_size]]

        loss = F.mse_loss(pred, target)

        total_loss += loss

    total_loss.backward()
    optimizer.step()

    print(f"Epoch {epoch} | Loss {total_loss.item():.4f} | beta {model.beta.item():.4f}")

torch.save(pred_all.cpu(), "data/opinion_embeddings_final.pt")
print("Saved final opinion embeddings")