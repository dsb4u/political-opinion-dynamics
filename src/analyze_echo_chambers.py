import torch
import pandas as pd
import numpy as np
import numpy as np

print("Loading embeddings...")
emb = torch.load("data/opinion_embeddings_final.pt")
emb = emb.detach().cpu().numpy()

print("Loading dataset...")
df = pd.read_csv("data/clean_comments_2019_09.csv")

# comment index map
id_to_index = {cid: i for i, cid in enumerate(df["id"])}

reply_similarities = []

print("Computing reply pair similarities...")

for _, row in df.iterrows():

    parent = row["parent_id"]

    if parent.startswith("t1_"):

        parent_id = parent[3:]

        if parent_id in id_to_index:

            i = id_to_index[row["id"]]
            j = id_to_index[parent_id]

            if i != j:  # avoid self-comparison

                sim = np.linalg.norm(emb[i] - emb[j])

                reply_similarities.append(sim)

reply_similarities = reply_similarities[:20000]

reply_mean = np.mean(reply_similarities)

print("Reply distance:", reply_mean)


print("\nComputing random pair similarities...")

n = len(reply_similarities)

rand_idx1 = np.random.randint(0, len(emb), n)
rand_idx2 = np.random.randint(0, len(emb), n)

# compute random similarities (pairwise)
rand_similarities = []

for i, j in zip(rand_idx1, rand_idx2):
    sim = np.linalg.norm(emb[i] - emb[j])

    rand_similarities.append(sim)

rand_mean = np.mean(rand_similarities)

print("Random distance:", rand_mean)

print("\nEcho chamber strength (distance):", rand_mean - reply_mean)