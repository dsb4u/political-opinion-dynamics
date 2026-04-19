import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import umap

print("Loading memory snapshots...")

mem1 = torch.load("data/memory_epoch3_step20000.pt").detach().cpu().numpy()
mem2 = torch.load("data/memory_epoch3_step60000.pt").detach().cpu().numpy()
mem3 = torch.load("data/memory_epoch3_step100000.pt").detach().cpu().numpy()


num_nodes = min(5000, mem1.shape[0])
mem1 = mem1[:num_nodes]
mem2 = mem2[:num_nodes]
mem3 = mem3[:num_nodes]

print("Running UMAP...")

reducer = umap.UMAP(n_components=2, random_state=42)
emb1_2d = reducer.fit_transform(mem1)
emb2_2d = reducer.transform(mem2)
emb3_2d = reducer.transform(mem3)

print("Clustering...")

k = 6
kmeans = KMeans(n_clusters=k, random_state=42)
labels = kmeans.fit_predict(mem1)   

print("Plotting...")

plt.figure(figsize=(18, 5))


plt.subplot(1, 3, 1)
plt.scatter(emb1_2d[:,0], emb1_2d[:,1], c=labels, s=5)
plt.title("Early (20k)")


plt.subplot(1, 3, 2)
plt.scatter(emb2_2d[:,0], emb2_2d[:,1], c=labels, s=5)
plt.title("Mid (60k)")


plt.subplot(1, 3, 3)
plt.scatter(emb3_2d[:,0], emb3_2d[:,1], c=labels, s=5)
plt.title("Late (100k)")

plt.tight_layout()
plt.savefig("data/opinion_shift.png", dpi=300)
plt.show()