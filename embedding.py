from sklearn.datasets import fetch_20newsgroups

print("Loading dataset...")

data = fetch_20newsgroups(subset="all")

documents = data.data

print("Total documents:", len(documents))


def clean_text(text):

    lines = text.split("\n")

    filtered = []

    for line in lines:
        if line.startswith(">"):
            continue
        if "@" in line:
            continue
        filtered.append(line)

    return " ".join(filtered)


documents = [clean_text(doc) for doc in documents]


from sentence_transformers import SentenceTransformer

print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Generating embeddings...")

embeddings = model.encode(
    documents,
    show_progress_bar=True
)

print("Embedding shape:", embeddings.shape)


import faiss
import numpy as np

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(np.array(embeddings))

print("Vector database built with", index.ntotal, "vectors")