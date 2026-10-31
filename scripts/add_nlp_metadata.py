"""Add topic and sentiment metadata to the ChromaDB chunks (re-uses the stored embeddings)."""
import chromadb
import numpy as np
import pandas as pd

from src.vector_store import get_collection


def main():
    docs = pd.read_parquet("data/processed/clean_corpus.parquet")[["id", "primary_category"]]
    nlp = pd.read_parquet("data/processed/doc_nlp.parquet")
    chunks = pd.read_parquet("data/processed/chunks.parquet")
    emb = np.load("data/processed/embeddings.npy")
    assert len(chunks) == len(emb), "chunks and embeddings do not match"

    meta = docs.merge(nlp, on="id").set_index("id")
    assert len(meta) == len(docs), "some documents lost in the merge"
    chunks = chunks.join(meta, on="doc_id")
    assert chunks["topic_id"].notna().all(), "some chunks have no topic metadata"

    collection = get_collection(chromadb.PersistentClient(path="chroma_db"))
    batch = 1000
    for start in range(0, len(chunks), batch):
        part = chunks.iloc[start:start + batch]
        collection.upsert(
            ids=part["chunk_id"].tolist(),
            documents=part["text"].tolist(),
            embeddings=emb[start:start + batch].tolist(),
            metadatas=[{
                "doc_id": r.doc_id, "chunk_index": int(r.chunk_index), "title": r.title,
                "categories": r.categories, "primary_category": r.primary_category,
                "url": r.url, "topic_id": int(r.topic_id), "topic_label": r.topic_label,
                "topic_strict": bool(r.topic_strict), "sentiment": r.sentiment,
                "sentiment_score": float(r.sentiment_score),
            } for r in part.itertuples()],
        )

    assert collection.count() == len(chunks), "count changed after upsert"
    print(f"Collection now has {collection.count()} chunks with topic and sentiment metadata")
    for name, where in [("topic 2", {"topic_id": 2}),
                        ("sentiment negative", {"sentiment": "negative"}),
                        ("cs.CV", {"primary_category": "cs.CV"})]:
        print(f"  chunks matching {name}: {len(collection.get(where=where, include=[])['ids'])}")


if __name__ == "__main__":
    main()