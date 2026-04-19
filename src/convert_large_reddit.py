import pandas as pd
import json

def process_file(input_path, output_path, max_lines=None):
    print(f"Processing {input_path}...")

    chunk_size = 100000   # process in chunks
    rows = []
    total = 0

    with open(input_path, "r") as f:
        for i, line in enumerate(f):
            try:
                data = json.loads(line)

                rows.append({
                    "id": data.get("id"),
                    "parent_id": data.get("parent_id"),
                    "created_utc": data.get("created_utc"),
                    "body": data.get("body"),
                    "subreddit": data.get("subreddit")
                })

                
                if len(rows) >= chunk_size:
                    df = pd.DataFrame(rows)
                    df.to_csv(output_path, mode='a', index=False, header=(total == 0))
                    total += len(rows)
                    rows = []
                    print(f"Processed {total} rows...")

                if max_lines and i >= max_lines:
                    break

            except:
                continue

    
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(output_path, mode='a', index=False, header=(total == 0))
        total += len(rows)

    print(f"Done {output_path} | Total rows: {total}")


#  RUN FOR EACH MONTH 

process_file("data/comments_2019-09", "data/comments_2019_09.csv")
process_file("data/comments_2019-10", "data/comments_2019_10.csv")
process_file("data/comments_2019-11", "data/comments_2019_11.csv")