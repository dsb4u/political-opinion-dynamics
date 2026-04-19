import torch
import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.cluster import KMeans

print("Loading embeddings...")
emb = torch.load("data/opinion_embeddings_final.pt")
emb = emb.detach().cpu().numpy()

print("Running UMAP...")

reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
coords = reducer.fit_transform(emb)

print("Clustering...")

kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(coords)

print("Cluster sizes:", np.bincount(labels))
print("Cluster percentages:", np.bincount(labels) / len(labels))

print("Plotting...")

plt.figure(figsize=(8,6))

plt.scatter(
    coords[:,0],
    coords[:,1],
    c=labels,
    s=1,
    cmap='viridis',
    alpha=0.6
)

plt.title("Opinion Space (UMAP Projection with Clusters)")
plt.xlabel("Component 1")
plt.ylabel("Component 2")

plt.tight_layout()
plt.show()