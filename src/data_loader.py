import pandas as pd

print("Loading dataset...")

# load first 200k comments instead of random sampling
df = pd.read_json(
    "data/comments_2019-09.bz2",
    lines=True,
    compression="bz2",
    nrows=200000
)

print("Loaded rows:", df.shape)

# remove deleted users
df = df[df["author"] != "[deleted]"]

# remove missing text
df = df[df["body_cleaned"].notna()]

# keep English comments only
df = df[df["language"] == "en"]

print("After cleaning:", df.shape)

# keep only required columns
df = df[
[
"id",
"parent_id",
"body_cleaned",
"author",
"created_utc",
"score",
"controversiality",
"subreddit"
]
]

print("Final dataset:", df.shape)

print(df.head())

# save cleaned dataset
df.to_csv("data/clean_comments_2019_09.csv", index=False)

print("Clean dataset saved to data/clean_comments_2019_09.csv")