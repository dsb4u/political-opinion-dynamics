import pandas as pd
import torch
from transformers import RobertaTokenizer, RobertaModel
from tqdm import tqdm

print("Loading cleaned dataset...")

df = pd.read_csv("data/clean_comments_2019_09.csv")

texts = df["body_cleaned"].tolist()

print("Number of comments:", len(texts))


tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaModel.from_pretrained("roberta-base")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

if device.type == "cuda":
    model = model.half()
    
model.eval()

embeddings = []

batch_size = 16

print("Generating embeddings...")

for i in tqdm(range(0, len(texts), batch_size)):

    batch = texts[i:i+batch_size]

    inputs = tokenizer(
        batch,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model(**inputs)

    cls_embeddings = outputs.last_hidden_state[:,0,:]

    embeddings.append(cls_embeddings.cpu())

embeddings = torch.cat(embeddings)

print("Embedding shape:", embeddings.shape)

torch.save(embeddings, "data/comment_embeddings.pt")

print("Embeddings saved to data/comment_embeddings.pt")