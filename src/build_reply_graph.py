import pandas as pd
import networkx as nx

print("Loading cleaned dataset...")

df = pd.read_csv("data/clean_comments_2019_09.csv")

print("Dataset size:", df.shape)

# remove t1_ prefix
df["parent_id"] = df["parent_id"].str.replace("t1_", "", regex=False)

print("Building reply graph...")

G = nx.DiGraph()

# add nodes
for comment_id in df["id"]:
    G.add_node(comment_id)

# add edges
valid_ids = set(df["id"])

for _, row in df.iterrows():
    
    parent = row["parent_id"]
    child = row["id"]

    if parent in valid_ids:
        G.add_edge(parent, child)

print("Graph created")

print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())