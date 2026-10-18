"""ChromaDB helpers: collection setup, ingestion and semantic search."""
import numpy as np

COLLECTION_NAME = "arxiv_chunks"


def get_collection(client, name=COLLECTION_NAME):
    return client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})


def ingest(collection, chunks, embeddings, batch_size=1000):
    """Upsert chunks + embeddings in batches. Safe to run more than once."""
    if len(chunks) != len(embeddings):
        raise ValueError(f"{len(chunks)} chunks but {len(embeddings)} embeddings")
    if not chunks["chunk_id"].is_unique:
        raise ValueError("chunk_id values must be unique")
    if (chunks["text"].str.strip() == "").any():
        raise ValueError("empty chunk text found")

    for start in range(0, len(chunks), batch_size):
        batch = chunks.iloc[start:start + batch_size]
        collection.upsert(
            ids=batch["chunk_id"].tolist(),
            documents=batch["text"].tolist(),
            embeddings=embeddings[start:start + batch_size].tolist(),
            metadatas=[
                {
                    "doc_id": r.doc_id,
                    "chunk_index": int(r.chunk_index),
                    "title": r.title,
                    "categories": r.categories,
                    "primary_category": r.primary_category,
                    "url": r.url,
                }
                for r in batch.itertuples()
            ],
        )


def search_by_embedding(collection, query_embedding, k=5, where=None):
    """Return the k closest chunks (closest first). Handles empty DB and k > size."""
    total = collection.count()
    if total == 0:
        return []
    res = collection.query(
        query_embeddings=[np.asarray(query_embedding).tolist()],
        n_results=min(k, total),
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    return [
        {
            "chunk_id": cid,
            "text": doc,
            "distance": dist,
            "similarity": 1 - dist,
            **meta,
        }
        for cid, doc, meta, dist in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
        )
    ]


def search(collection, model, query, k=5, where=None):
    """Embed a text query and search."""
    if not query or not query.strip():
        raise ValueError("query must not be empty")
    emb = model.encode(query.strip(), normalize_embeddings=True)
    return search_by_embedding(collection, emb, k, where)