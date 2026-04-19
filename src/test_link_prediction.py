import torch
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

import random

np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

print("Loading embeddings...")
emb = torch.load("data/tgn_embeddings_best.pt").detach().cpu()

print("Loading dataset...")
df = pd.read_csv("data/clean_comments_2019_09_11.csv")

import pickle

print("Loading id_map...")
with open("data/id_map.pkl", "rb") as f:
    id_map = pickle.load(f)

# apply same filtering as training
df["parent_clean"] = df["parent_id"].str.replace("t1_", "", regex=False)
df["parent_clean"] = df["parent_clean"].str.replace("t3_", "", regex=False)

df = df[df["parent_clean"].isin(id_map)]

df["src"] = df["parent_clean"].map(id_map)
df["dst"] = df["id"].map(id_map)

df = df.dropna(subset=["src", "dst"])






# convert to int
df["src"] = df["src"].astype(int)
df["dst"] = df["dst"].astype(int)

print("Preparing test samples...")
print("Number of edges:", len(df))

# sample real edges
real_edges = df.sample(30000)

# sample fake edges
num_nodes = emb.shape[0]
df = df[(df["src"] < num_nodes) & (df["dst"] < num_nodes)]
fake_src = np.random.randint(0, num_nodes, 5000)
fake_dst = np.random.randint(0, num_nodes, 5000)


# same predictor as training
predictor = torch.nn.Sequential(
    torch.nn.Linear(512, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 1)
)

predictor.load_state_dict(torch.load("data/link_predictor_best.pt"))
predictor.eval()


def score(u, v):
    u_emb = emb[u]
    v_emb = emb[v]
    x = torch.cat([u_emb, v_emb], dim=0)
    return predictor(x).item()

real_scores = [score(row["src"], row["dst"]) for _, row in real_edges.iterrows()]
fake_scores = [score(u, v) for u, v in zip(fake_src, fake_dst)]

y_true = [1]*len(real_scores) + [0]*len(fake_scores)
y_scores = real_scores + fake_scores

auc = roc_auc_score(y_true, y_scores)

print(f"Link Prediction AUC: {auc:.4f}")