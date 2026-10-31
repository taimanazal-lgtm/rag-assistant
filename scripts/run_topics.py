"""Discover corpus topics with BERTopic, reusing the chunk embeddings from Week 2."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from bertopic import BERTopic
from bertopic.dimensionality import BaseDimensionalityReduction
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

DOCS_PATH = "data/processed/clean_corpus.parquet"
CHUNKS_PATH = "data/processed/chunks.parquet"
EMB_PATH = "data/processed/embeddings.npy"
UMAP_PATH = Path("data/processed/doc_umap.npy")
TOPICS_PATH = "data/processed/doc_topics.parquet"
REVIEW_PATH = Path("results/topic_review.csv")
REPORT_PATH = Path("results/topics_report.json")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MIN_CLUSTER_SIZE = 30
MIN_SAMPLES = 5
SEED = 42


def doc_embeddings(docs):
    """Average the chunk embeddings of each document (uses all of its text)."""
    chunks = pd.read_parquet(CHUNKS_PATH)
    emb = np.load(EMB_PATH)
    mean = pd.DataFrame(emb).groupby(chunks["doc_id"].to_numpy()).mean()
    out = mean.loc[docs["id"]].to_numpy()
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def reduced_embeddings(doc_emb):
    """UMAP is the slow step (about 75 s), so compute it once and cache it."""
    if UMAP_PATH.exists():
        cached = np.load(UMAP_PATH)
        if len(cached) == len(doc_emb):
            return cached
    reduced = UMAP(n_neighbors=15, n_components=5, min_dist=0.0,
                   metric="cosine", random_state=SEED).fit_transform(doc_emb)
    np.save(UMAP_PATH, reduced)
    return reduced


def main():
    docs = pd.read_parquet(DOCS_PATH)
    doc_emb = doc_embeddings(docs)
    assert doc_emb.shape[0] == len(docs), "embedding rows != documents"
    reduced = reduced_embeddings(doc_emb)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    topic_model = BERTopic(
        embedding_model=SentenceTransformer(EMBED_MODEL, device=device),
        umap_model=BaseDimensionalityReduction(),  # UMAP was already applied above
        hdbscan_model=HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE, min_samples=MIN_SAMPLES,
                              metric="euclidean", cluster_selection_method="leaf"),
        # no min_df: the vectorizer sees one document per topic, so min_df would drop topic-specific words
        vectorizer_model=CountVectorizer(stop_words="english", ngram_range=(1, 2)),
        top_n_words=10,
        verbose=True,
    )
    topics, _ = topic_model.fit_transform(docs["text"].tolist(), embeddings=reduced)
    docs["topic"] = topics

    assigned = docs[docs["topic"] >= 0]
    sizes = assigned.groupby("topic").size().sort_values(ascending=False)
    outlier_share = float((docs["topic"] == -1).mean())

    rows = []
    for tid, size in sizes.items():
        sub = docs[docs["topic"] == tid]
        words = [w for w, _ in topic_model.get_topic(tid)][:8]
        titles = sub["title"].sample(min(3, len(sub)), random_state=SEED).tolist()
        cats = sub["primary_category"].value_counts(normalize=True)
        rows.append({
            "topic_id": int(tid),
            "size": int(size),
            "top_words": ", ".join(words),
            "dominant_category": f"{cats.index[0]} ({cats.iloc[0]:.0%})",
            "example_titles": " | ".join(titles),
            "coherence": "",  # to be filled by hand: good / mixed / bad
        })
    review = pd.DataFrame(rows)

    REVIEW_PATH.parent.mkdir(parents=True, exist_ok=True)
    review.to_csv(REVIEW_PATH, index=False, encoding="utf-8-sig")
    docs[["id", "topic"]].to_parquet(TOPICS_PATH, index=False)
    REPORT_PATH.write_text(json.dumps({
        "n_documents": len(docs),
        "n_topics": int(len(sizes)),
        "outlier_share": round(outlier_share, 4),
        "topic_size_min": int(sizes.min()),
        "topic_size_median": float(sizes.median()),
        "topic_size_max": int(sizes.max()),
        "min_cluster_size": MIN_CLUSTER_SIZE,
        "min_samples": MIN_SAMPLES,
        "cluster_selection_method": "leaf",
        "seed": SEED,
    }, indent=2), encoding="utf-8")

    print(f"\nTopics: {len(sizes)} | outliers (topic -1): {outlier_share:.1%}")
    print(f"Topic sizes: min {sizes.min()}, median {sizes.median():.0f}, max {sizes.max()}\n")
    for r in rows[:15]:
        print(f"[{r['topic_id']:>3}] n={r['size']:<4} {r['dominant_category']:<12} {r['top_words']}")
    print(f"\nSaved {REVIEW_PATH}, {TOPICS_PATH}, {REPORT_PATH}")


if __name__ == "__main__":
    main()