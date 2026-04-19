import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load data


print("Loading dataset...")
df = pd.read_csv("data/clean_comments_2019_09_11.csv")

df = df.sample(frac=0.1, random_state=42)

# map ids to indices
node_ids = df["id"].unique()
id_map = {nid: i for i, nid in enumerate(node_ids)}

import pickle

with open("data/id_map.pkl", "wb") as f:
    pickle.dump(id_map, f)

print("Saved id_map")


# clean parent_id


df["parent_clean"] = df["parent_id"].str.replace("t1_", "", regex=False)
df["parent_clean"] = df["parent_clean"].str.replace("t3_", "", regex=False)

# keep only valid interactions
df = df[df["parent_clean"].isin(id_map)]

# map to indices
df["src"] = df["parent_clean"].map(id_map)
df["dst"] = df["id"].map(id_map)

# sort by time
df = df.sort_values("created_utc")

print("Total interactions:", len(df))


# TGN Model


class TGN(nn.Module):
    def __init__(self, num_nodes, memory_dim=256):
        super().__init__()

        self.register_buffer("memory", torch.zeros(num_nodes, memory_dim))

        self.message_mlp = nn.Sequential(
            nn.Linear(memory_dim * 2 + 1, 128),
            nn.ReLU(),
            nn.Linear(128, memory_dim)
        )

        self.gru = nn.GRUCell(memory_dim, memory_dim)
        self.predictor = nn.Sequential(
            nn.Linear(memory_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, src, dst, time):

        src_mem = self.memory[src]
        dst_mem = self.memory[dst]

        time = time.view(-1, 1)

        message_input = torch.cat([src_mem, dst_mem, time], dim=1)
        message = self.message_mlp(message_input)

        
        new_src = self.gru(message, src_mem)
        new_dst = self.gru(message, dst_mem)

        
        self.memory[src] = new_src.detach()
        self.memory[dst] = new_dst.detach()

        edge_input = torch.cat([new_src, new_dst], dim=1)
        score = self.predictor(edge_input)

        return score, new_src, new_dst


# Training loop


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TGN(num_nodes=len(id_map)).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()

print("Starting training...")

# normalize entire time column
time_values = df["created_utc"].values.astype(float)

time_mean = time_values.mean()
time_std = time_values.std()

for epoch in range(4):

    total_loss = 0
    num_steps = 0

    for _, row in tqdm(df.iterrows(), total=len(df)):

        src = torch.tensor([row["src"]]).to(device)
        dst = torch.tensor([row["dst"]]).to(device)
        time = torch.tensor(
            [(row["created_utc"] - time_mean) / (time_std + 1e-8)],
            dtype=torch.float32
        ).to(device)

        time = time.view(-1, 1)

        # positive pair
        pos_score, src_emb, dst_emb = model(src, dst, time)
        # positive label
        pos_label = torch.ones(1, 1, device=device)

        
        # multiple negative samples
        num_neg = 8

        neg_nodes = torch.randint(0, len(id_map), (num_neg,), device=device)

        neg_scores = []
        for neg_node in neg_nodes:
            neg_node = neg_node.unsqueeze(0)
            neg_score, _, _ = model(src, neg_node, time)
            neg_scores.append(neg_score)

        neg_scores = torch.cat(neg_scores, dim=0)

        neg_labels = torch.zeros(num_neg, 1, device=device)

        # combine
        scores = torch.cat([pos_score, neg_scores], dim=0)
        labels = torch.cat([pos_label, neg_labels], dim=0)

        # BCE loss (link prediction objective)
        pos_weight = torch.tensor([num_neg], device=device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        loss = loss_fn(scores, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_steps += 1
        
        # SAVE MEMORY SNAPSHOTS
        
        if num_steps in [20000, 60000, 100000]:
            filename = f"data/memory_epoch{epoch}_step{num_steps}.pt"

            torch.save(
                model.memory.detach().cpu(),
                filename
            )

            print(f"Saved memory snapshot: {filename}")

    avg_loss = total_loss / num_steps
    print(f"Epoch {epoch} | Avg Loss {avg_loss:.4f}")

# save memory as embeddings
torch.save(model.memory.cpu(), "data/tgn_embeddings_best.pt")
torch.save(model.predictor.state_dict(), "data/link_predictor_best.pt")

print("Saved TGN embeddings")