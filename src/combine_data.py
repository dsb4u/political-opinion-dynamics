import pandas as pd

print("Loading cleaned data...")

df1 = pd.read_csv("data/clean_comments_2019_09.csv")
df2 = pd.read_csv("data/clean_comments_2019_10.csv")
df3 = pd.read_csv("data/clean_comments_2019_11.csv")

print("Combining...")
df = pd.concat([df1, df2, df3], ignore_index=True)

# convert to numeric
df["created_utc"] = pd.to_numeric(df["created_utc"], errors="coerce")

# drop invalid rows 
df = df.dropna(subset=["created_utc"])

print("Sorting by time...")
df = df.sort_values("created_utc")

print("Saving combined dataset...")
df.to_csv("data/clean_comments_2019_09_11.csv", index=False)

print("Done. Total rows:", len(df))