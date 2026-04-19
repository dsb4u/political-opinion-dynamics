import torch
import pandas as pd
import numpy as np

print("Loading embeddings...")
emb = torch.load("data/opinion_embeddings_final.pt")
emb = emb.detach().cpu().numpy()

print("Embedding shape:", emb.shape)

print("Loading dataset...")
df = pd.read_csv("data/clean_comments_2019_09.csv")

# convert timestamps
df["created_utc"] = pd.to_datetime(df["created_utc"], unit="s")

# same window definition used during training
df["time_window"] = df["created_utc"].dt.floor("6h")

windows = sorted(df["time_window"].unique())

polarization_scores = []

print("\nComputing polarization...")

for w in windows:

    idx = df[df["time_window"] == w].index
    vecs = emb[idx]

    # polarization = mean variance across embedding dimensions
    score = np.mean(np.var(vecs, axis=0))

    polarization_scores.append(score)

print("\nPolarization per time window:\n")

for w, p in zip(windows, polarization_scores):
    print(f"{w} : {p:.6f}")

import pandas as pd

results_df = pd.DataFrame({
    "time_window": windows,
    "polarization": polarization_scores
})

results_df.to_csv("results/polarization.csv", index=False)

print("\nSaved results to results/polarization.csv")