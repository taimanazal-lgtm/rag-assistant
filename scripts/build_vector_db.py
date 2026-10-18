"""Load chunks + embeddings into a persistent ChromaDB."""
import time

import chromadb
import numpy as np
import pandas as pd

from src.vector_store import COLLECTION_NAME, get_collection, ingest

DOCS_PATH = "data/processed/clean_corpus.parquet"
CHUNKS_PATH = "data/processed/chunks.parquet"
EMB_PATH = "data/processed/embeddings.npy"
DB_PATH = "chroma_db"


def main():
    docs = pd.read_parquet(DOCS_PATH)
    chunks = pd.read_parquet(CHUNKS_PATH)
    embeddings = np.load(EMB_PATH)

    # primary_category lives in the docs table: add it to each chunk (row order unchanged)
    chunks["primary_category"] = chunks["doc_id"].map(docs.set_index("id")["primary_category"])
    assert chunks["primary_category"].notna().all(), "Missing primary_category"

    client = chromadb.PersistentClient(path=DB_PATH)
    collection = get_collection(client)

    start = time.perf_counter()
    ingest(collection, chunks, embeddings)
    print(f"Ingested in {time.perf_counter() - start:.1f}s")

    # validation
    assert collection.count() == len(chunks), "Count mismatch after ingestion"
    sample = collection.get(ids=[chunks["chunk_id"].iloc[0]], include=["documents", "metadatas"])
    print(f"Collection '{COLLECTION_NAME}': {collection.count()} chunks")
    print("Sample metadata:", sample["metadatas"][0])


if __name__ == "__main__":
    main()