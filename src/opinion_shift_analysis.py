import torch
import numpy as np
from sklearn.cluster import KMeans

print("Loading memory snapshots...")

mem1 = torch.load("data/memory_epoch3_step20000.pt")
mem2 = torch.load("data/memory_epoch3_step60000.pt")
mem3 = torch.load("data/memory_epoch3_step100000.pt")

mem1 = mem1.detach().cpu().numpy()
mem2 = mem2.detach().cpu().numpy()
mem3 = mem3.detach().cpu().numpy()


# EMBEDDING DRIFT


print("\n=== Embedding Drift ===")

drift_1_2 = np.linalg.norm(mem2 - mem1, axis=1)
drift_2_3 = np.linalg.norm(mem3 - mem2, axis=1)

print("Avg drift (early → mid):", drift_1_2.mean())
print("Avg drift (mid → late):", drift_2_3.mean())


# CLUSTER TRANSITIONS


print("\n=== Cluster Transitions ===")

k = 6

kmeans1 = KMeans(n_clusters=k, random_state=42)
labels1 = kmeans1.fit_predict(mem1)

kmeans2 = KMeans(n_clusters=k, random_state=42)
labels2 = kmeans2.fit_predict(mem2)

kmeans3 = KMeans(n_clusters=k, random_state=42)
labels3 = kmeans3.fit_predict(mem3)

# Compare via distances
center_shift_1_2 = np.linalg.norm(kmeans1.cluster_centers_ - kmeans2.cluster_centers_, axis=1).mean()
center_shift_2_3 = np.linalg.norm(kmeans2.cluster_centers_ - kmeans3.cluster_centers_, axis=1).mean()

print("Cluster center shift (early → mid):", center_shift_1_2)
print("Cluster center shift (mid → late):", center_shift_2_3)


# TEMPORAL SMOOTHNESS


print("\n=== Temporal Smoothness ===")

smooth_1_2 = drift_1_2.mean()
smooth_2_3 = drift_2_3.mean()

print("Temporal change (early → mid):", smooth_1_2)
print("Temporal change (mid → late):", smooth_2_3)

if smooth_2_3 < smooth_1_2:
    print("Trend: embeddings stabilizing over time")
else:
    print("Trend: embeddings becoming more dynamic over time")