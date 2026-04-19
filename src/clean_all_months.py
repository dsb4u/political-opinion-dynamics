import pandas as pd

def clean_file(input_path, output_path):
    print(f"Cleaning {input_path}...")

    df = pd.read_csv(input_path)

    # remove missing critical fields
    df = df.dropna(subset=["id", "parent_id", "created_utc"])

    # remove deleted/removed
    df = df[df["body"] != "[deleted]"]
    df = df[df["body"] != "[removed]"]

    # keep only needed columns 
    df = df[["id", "parent_id", "created_utc"]]

    df.to_csv(output_path, index=False)
    print(f"Saved {output_path} | Rows: {len(df)}")


clean_file("data/comments_2019_09.csv", "data/clean_comments_2019_09.csv")
clean_file("data/comments_2019_10.csv", "data/clean_comments_2019_10.csv")
clean_file("data/comments_2019_11.csv", "data/clean_comments_2019_11.csv")