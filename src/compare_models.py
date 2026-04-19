import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import pairwise_distances

def load_embeddings(path):
    emb = torch.load(path)
    return emb.detach().cpu().numpy()

def compute_metrics(emb, k=6):
    # KMeans
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(emb)

    # Silhouette 
    sil = silhouette_score(emb, labels)

    # Distance matrix 
    dists = pairwise_distances(emb)

    intra = []
    inter = []

    for i in range(len(emb)):
        same_cluster = labels == labels[i]
        diff_cluster = labels != labels[i]

        # avoid self-distance
        intra_dist = dists[i][same_cluster]
        intra_dist = intra_dist[intra_dist > 0]

        inter_dist = dists[i][diff_cluster]

        if len(intra_dist) > 0:
            intra.append(np.mean(intra_dist))
        if len(inter_dist) > 0:
            inter.append(np.mean(inter_dist))

    intra_mean = np.mean(intra)
    inter_mean = np.mean(inter)
    ratio = inter_mean / intra_mean

    return sil, intra_mean, inter_mean, ratio

print("Loading embeddings...")

models = {
    "RoBERTa": "data/opinion_embeddings_final.pt",
    "GAT": "data/comment_embeddings.pt",
    "TGN": "data/tgn_embeddings_best.pt"
}

for name, path in models.items():
    print(f"\n=== {name} ===")

    emb = load_embeddings(path)

    
    if emb.shape[0] > 5000:
        emb = emb[:5000]

    sil, intra, inter, ratio = compute_metrics(emb)

    print(f"Silhouette Score: {sil:.4f}")
    print(f"Intra-cluster Distance: {intra:.4f}")
    print(f"Inter-cluster Distance: {inter:.4f}")
    print(f"Inter/Intra Ratio: {ratio:.4f}")