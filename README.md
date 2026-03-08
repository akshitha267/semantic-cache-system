# Semantic Search System with Fuzzy Clustering and Semantic Cache

This project implements a lightweight semantic search system over the **20 Newsgroups dataset**.  
The system demonstrates how embeddings, vector databases, clustering, and semantic caching can be combined to build an efficient semantic retrieval pipeline.

---

## Overview

Traditional search systems rely on keyword matching, which treats queries with similar meaning but different wording as unrelated.

This system instead performs **semantic search** by converting text into dense vector embeddings and retrieving documents based on similarity in vector space.

Additionally, the system implements a **semantic cache** capable of detecting when two queries are semantically similar and reusing previously computed results.

---

## Dataset

The system uses the **20 Newsgroups dataset**, a collection of approximately 20,000 documents across 20 different topic categories including:

- Space
- Politics
- Religion
- Computers
- Electronics
- Sports
- Firearms

The dataset contains informal newsgroup posts, which often include overlapping topics, making it suitable for demonstrating **fuzzy clustering**.

Dataset source:
https://archive.ics.uci.edu/dataset/113/twenty+newsgroups

---

## System Architecture
User Query
↓
Query Embedding
↓
Semantic Cache Lookup
↓
Vector Database Search (FAISS)
↓
Document Retrieval
↓
Cluster Prediction
↓
Return Result


---

## Technologies Used

| Component | Technology |
|--------|--------|
Embeddings | SentenceTransformers (all-MiniLM-L6-v2) |
Vector Database | FAISS |
Clustering | Gaussian Mixture Model |
Similarity Metric | Cosine Similarity |
API Framework | FastAPI |
Server | Uvicorn |

---

## Key Components

### Embedding Generation

Documents are converted into dense semantic embeddings using the **SentenceTransformer model `all-MiniLM-L6-v2`**.

These embeddings capture semantic meaning and allow similarity comparisons between documents and queries.

---

### Vector Database

All document embeddings are stored in a **FAISS index**, enabling efficient nearest-neighbour search in high-dimensional vector space.

This allows the system to retrieve documents based on semantic similarity instead of keyword matching.

---

### Fuzzy Clustering

The system uses a **Gaussian Mixture Model (GMM)** to cluster document embeddings.

Unlike traditional clustering methods, GMM produces **probability distributions over clusters**, allowing documents to belong to multiple clusters with different probabilities.

This better reflects the nature of real-world text where topics often overlap.

---

### Semantic Cache

A custom semantic cache is implemented to avoid redundant computation.

When a query is received:

1. The query is converted into an embedding
2. Cosine similarity is computed between the query embedding and cached query embeddings
3. If similarity exceeds a threshold, the cached result is returned
4. Otherwise, a new vector search is performed and the result is stored in the cache

This mechanism allows the system to reuse results for semantically similar queries.

---

## API Endpoints

### POST /query

Accepts a natural language query and returns the most relevant document.

Example request:

```json
{
  "query": "space shuttle launch"
}

