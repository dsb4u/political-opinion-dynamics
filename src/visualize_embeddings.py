import torch
import umap
import matplotlib.pyplot as plt

# change file name depending on model
embedding_file = "data/tgn_embeddings_best.pt"
# options:
# "data/opinion_embeddings_final.pt"
# "data/comment_embeddings.pt"
# "data/tgn_embeddings_best.pt"

print("Loading embeddings...")
emb = torch.load(embedding_file).detach().cpu().numpy()


emb = emb[:50000]

print("Running UMAP...")
reducer = umap.UMAP(n_components=2, n_neighbors=10, min_dist=0.3, random_state=42)
emb_2d = reducer.fit_transform(emb)

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=5, random_state=42)
labels = kmeans.fit_predict(emb)
print("Plotting...")
plt.figure(figsize=(8, 6))
plt.scatter(emb_2d[:, 0], emb_2d[:, 1], s=2, alpha=0.6)
plt.title(f"UMAP Projection: {embedding_file}")
plt.savefig(embedding_file.replace(".pt", "_umap.png"))
plt.show()