import pandas as pd
import networkx as nx
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import community as community_louvain
from sklearn.metrics import pairwise_distances
import numpy as np


#GRAPH + MODULARITY


print("Loading dataset...")
df = pd.read_csv("data/clean_comments_2019_09_11.csv")


df["parent_clean"] = df["parent_id"].str.replace("t1_", "", regex=False)
df["parent_clean"] = df["parent_clean"].str.replace("t3_", "", regex=False)

df = df.dropna(subset=["parent_clean", "id"])


df = df.head(200000)

print("Building graph...")
G = nx.Graph()

edges = list(zip(df["parent_clean"], df["id"]))
G.add_edges_from(edges)

print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())

# Community detection
print("Running Louvain...")
partition = community_louvain.best_partition(G)

num_communities = len(set(partition.values()))
print("Number of communities:", num_communities)

# Modularity
modularity = community_louvain.modularity(partition, G)
print("Modularity:", modularity)

print("\nComputing echo chamber score...")

internal_edges = 0
external_edges = 0

for u, v in G.edges():
    if partition[u] == partition[v]:
        internal_edges += 1
    else:
        external_edges += 1

print("Internal edges:", internal_edges)
print("External edges:", external_edges)

if external_edges > 0:
    echo_score = internal_edges / external_edges
else:
    echo_score = float('inf')

print("Echo Chamber Score:", echo_score)


# EDGE DENSITY ANALYSIS


print("\nComputing edge density...")

# Overall graph density
overall_density = nx.density(G)
print("Overall Graph Density:", overall_density)

# average community density
from collections import defaultdict

communities = defaultdict(list)
for node, comm in partition.items():
    communities[comm].append(node)

densities = []

for comm_nodes in communities.values():
    if len(comm_nodes) > 1:
        subgraph = G.subgraph(comm_nodes)
        densities.append(nx.density(subgraph))

if len(densities) > 0:
    avg_density = sum(densities) / len(densities)
else:
    avg_density = 0

print("Average Community Density:", avg_density)


#SILHOUETTE SCORE


print("\nLoading embeddings...")
emb = torch.load("data/tgn_embeddings_best.pt").detach().cpu().numpy()


emb = emb[:30000]

print("Running KMeans...")
print("Running KMeans (multiple cluster settings)...")

best_score = -1
best_k = None

for k in [4, 6, 8, 10]:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(emb)
    
    score = silhouette_score(emb, labels)
    print(f"k={k} → silhouette={score:.4f}")
    
    if score > best_score:
        best_score = score
        best_k = k

print("\nBest K:", best_k)
print("Best Silhouette Score:", best_score)
labels = kmeans.fit_predict(emb)

print("Computing silhouette score...")
score = silhouette_score(emb, labels)

print("Silhouette Score:", score)


# TGN CLUSTER QUALITY ANALYSIS


print("\nAnalyzing cluster quality (safe version)...")

# sample subset
sample_size = 5000
idx = np.random.choice(len(emb), sample_size, replace=False)

emb_sample = emb[idx]
labels_sample = labels[idx]

from sklearn.metrics import pairwise_distances

dist_matrix = pairwise_distances(emb_sample)

intra_dist = []
inter_dist = []

for i in range(sample_size):
    for j in range(i+1, sample_size):
        if labels_sample[i] == labels_sample[j]:
            intra_dist.append(dist_matrix[i][j])
        else:
            inter_dist.append(dist_matrix[i][j])

avg_intra = np.mean(intra_dist)
avg_inter = np.mean(inter_dist)

print("Average Intra-cluster Distance:", avg_intra)
print("Average Inter-cluster Distance:", avg_inter)
print("Inter/Intra Ratio:", avg_inter / avg_intra)