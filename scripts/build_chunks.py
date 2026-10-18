"""Chunk the clean corpus and save data/processed/chunks.parquet"""
import pandas as pd

from src.chunking import chunk_documents

IN_PATH = "data/processed/clean_corpus.parquet"
OUT_PATH = "data/processed/chunks.parquet"
CHUNK_SIZE = 800
OVERLAP = 200


def main():
    docs = pd.read_parquet(IN_PATH)
    chunks = chunk_documents(docs, CHUNK_SIZE, OVERLAP)
    chunks["n_chars"] = chunks["text"].str.len()

    # validation
    assert chunks["chunk_id"].is_unique, "Duplicate chunk ids"
    assert chunks["doc_id"].nunique() == len(docs), "Some documents got no chunks"
    assert (chunks["n_chars"] <= CHUNK_SIZE).all(), "Chunk longer than CHUNK_SIZE"
    assert (chunks["n_chars"] > 0).all(), "Empty chunk found"
    pairs = chunks.groupby("doc_id")["text"].apply(list)
    n_pairs = n_overlap = 0
    for texts in pairs:
        for a, b in zip(texts, texts[1:]):
            n_pairs += 1
            n_overlap += b[:40] in a
    print(f"Chunk pairs with real overlap: {n_overlap}/{n_pairs}")
    print(f"Documents: {len(docs)} -> Chunks: {len(chunks)}")
    print("\nChunks per document:")
    print(chunks.groupby("doc_id").size().describe())
    print("\nChunk length (chars):")
    print(chunks["n_chars"].describe())
    print(f"\nVery short chunks (< 150 chars): {(chunks['n_chars'] < 150).sum()}")

    # look at one document's chunks
    sample = chunks[chunks["doc_id"] == chunks["doc_id"].iloc[0]]
    for _, row in sample.iterrows():
        print(f"\n--- {row['chunk_id']} ({row['n_chars']} chars) ---")
        print(row["text"])

    chunks.to_parquet(OUT_PATH, index=False)
    print(f"\nSaved {len(chunks)} chunks to {OUT_PATH}")


if __name__ == "__main__":
    main()