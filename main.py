from sklearn.datasets import fetch_20newsgroups

print("Loading dataset...")

data = fetch_20newsgroups(subset="train")

documents = data.data[:2000]  # keep small so it runs fast

print("Total documents loaded:", len(documents))

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

from embedding import model, documents, embeddings, index
from clustering import build_clusters
from cache import SemanticCache

app = FastAPI()

cache = SemanticCache()
gmm, cluster_probs = build_clusters(embeddings)


class QueryRequest(BaseModel):
    query: str


@app.post("/query")
def query_system(req: QueryRequest):

    query = req.query

    q_embed = model.encode([query])[0]

    hit, entry, sim = cache.lookup(q_embed)

    if hit:

        return {
            "query": query,
            "cache_hit": True,
            "matched_query": entry["query"],
            "similarity_score": float(sim),
            "result": entry["result"],
            "dominant_cluster": entry["cluster"]
        }

    D, I = index.search(np.array([q_embed]), 3)

    result = documents[I[0][0]]

    cluster = int(gmm.predict([q_embed])[0])

    cache.store(query, q_embed, result, cluster)

    return {
        "query": query,
        "cache_hit": False,
        "matched_query": None,
        "similarity_score": None,
        "result": result,
        "dominant_cluster": cluster
    }


@app.get("/cache/stats")
def stats():

    return cache.stats()


@app.delete("/cache")
def clear():

    cache.clear()

    return {"message": "cache cleared"}


