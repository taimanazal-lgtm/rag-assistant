"""Benchmark semantic search: latency per query type, per k, plus a sanity check."""
import json
import time
from pathlib import Path

import chromadb
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from src.vector_store import get_collection, search_by_embedding

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNKS_PATH = "data/processed/chunks.parquet"
EMB_PATH = "data/processed/embeddings.npy"
DB_PATH = "chroma_db"
OUT_PATH = Path("results/retrieval_benchmark.json")
REPEATS = 20
K = 5

QUERIES = [
    {"type": "short keyword", "query": "transformers", "where": None},
    {"type": "short keyword", "query": "reinforcement learning", "where": None},
    {"type": "question", "query": "How do large language models handle long context windows?", "where": None},
    {"type": "question", "query": "What methods reduce hallucination in language models?", "where": None},
    {"type": "question", "query": "How are diffusion models used for image generation?", "where": None},
    {"type": "question", "query": "How can neural networks be made robust against adversarial attacks?", "where": None},
    {"type": "long query", "query": "We study how retrieval-augmented generation improves the factual accuracy of question answering systems by grounding answers in retrieved documents", "where": None},
    {"type": "filtered (cs.CV)", "query": "object detection with transformers", "where": {"primary_category": "cs.CV"}},
    {"type": "filtered (cs.CL)", "query": "object detection with transformers", "where": {"primary_category": "cs.CL"}},
    {"type": "off-topic", "query": "best recipe for chocolate cake", "where": None},
]


def stats(values):
    return {
        "mean_ms": round(float(np.mean(values)), 2),
        "p50_ms": round(float(np.percentile(values, 50)), 2),
        "p95_ms": round(float(np.percentile(values, 95)), 2),
        "max_ms": round(float(np.max(values)), 2),
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(MODEL_NAME, device=device)
    collection = get_collection(chromadb.PersistentClient(path=DB_PATH))
    print(f"Device: {device} | chunks in DB: {collection.count()}\n")

    # ---- 1. latency per query type ----
    report = []
    for q in QUERIES:
        emb = model.encode(q["query"], normalize_embeddings=True)  # warm-up
        search_by_embedding(collection, emb, K, q["where"])

        embed_ms, search_ms = [], []
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            emb = model.encode(q["query"], normalize_embeddings=True)
            t1 = time.perf_counter()
            results = search_by_embedding(collection, emb, K, q["where"])
            t2 = time.perf_counter()
            embed_ms.append((t1 - t0) * 1000)
            search_ms.append((t2 - t1) * 1000)

        total = np.array(embed_ms) + np.array(search_ms)
        report.append({
            **q,
            "embed": stats(embed_ms),
            "search": stats(search_ms),
            "total": stats(total),
            "top_results": [
                {"title": r["title"], "similarity": round(r["similarity"], 3),
                 "category": r["primary_category"], "url": r["url"]}
                for r in results[:3]
            ],
        })

    print(f"{'type':<17}{'query':<34}{'embed':>8}{'search':>8}{'total p50':>10}{'total p95':>10}")
    for r in report:
        print(f"{r['type']:<17}{r['query'][:32]:<34}{r['embed']['p50_ms']:>8}"
              f"{r['search']['p50_ms']:>8}{r['total']['p50_ms']:>10}{r['total']['p95_ms']:>10}")

    print("\nTop-3 results (check relevance by eye):")
    for r in report:
        print(f"\n[{r['type']}] {r['query'][:70]}")
        for t in r["top_results"]:
            print(f"   {t['similarity']:.3f} | {t['category']} | {t['title'][:70]}")

    # ---- 2. effect of k (search only) ----
    emb = model.encode(QUERIES[2]["query"], normalize_embeddings=True)
    k_report = {}
    print("\nSearch latency vs k:")
    for k in [1, 5, 10, 20, 50]:
        times = []
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            search_by_embedding(collection, emb, k)
            times.append((time.perf_counter() - t0) * 1000)
        k_report[k] = stats(times)
        print(f"   k={k:<3} p50={k_report[k]['p50_ms']} ms  p95={k_report[k]['p95_ms']} ms")

    # ---- 3. sanity check: a chunk must find itself ----
    chunks = pd.read_parquet(CHUNKS_PATH)
    embeddings = np.load(EMB_PATH)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(chunks), size=200, replace=False)
    hits = sum(
        search_by_embedding(collection, embeddings[i], 1)[0]["chunk_id"] == chunks["chunk_id"].iloc[i]
        for i in idx
    )
    self_recall = hits / len(idx)
    print(f"\nSelf-retrieval check: {hits}/{len(idx)} chunks found themselves at rank 1")
    assert self_recall >= 0.99, "Ids and embeddings look misaligned!"

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({
        "device": device, "n_chunks": collection.count(), "k": K, "repeats": REPEATS,
        "per_query": report, "latency_vs_k": k_report, "self_retrieval_recall_at_1": self_recall,
    }, indent=2), encoding="utf-8")
    print(f"\nSaved {OUT_PATH}")


if __name__ == "__main__":
    main()