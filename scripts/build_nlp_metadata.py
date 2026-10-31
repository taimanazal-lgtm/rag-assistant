"""Compute topic + sentiment metadata for every document."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.nlp import (assign_nearest_topic, doc_embeddings, label_from_score,
                     text_sentiment, topic_label)

DOCS_PATH = "data/processed/clean_corpus.parquet"
CHUNKS_PATH = "data/processed/chunks.parquet"
EMB_PATH = "data/processed/embeddings.npy"
TOPICS_PATH = "data/processed/doc_topics.parquet"
REVIEW_PATH = "results/topic_review.csv"
OUT_PATH = "data/processed/doc_nlp.parquet"
REPORT_PATH = Path("results/nlp_metadata_report.json")


def main():
    docs = pd.read_parquet(DOCS_PATH)
    strict = pd.read_parquet(TOPICS_PATH).set_index("id").loc[docs["id"], "topic"].to_numpy()
    emb = doc_embeddings(docs, pd.read_parquet(CHUNKS_PATH), np.load(EMB_PATH))
    filled, threshold = assign_nearest_topic(emb, strict)

    words = pd.read_csv(REVIEW_PATH).set_index("topic_id")["top_words"].to_dict()
    analyzer = SentimentIntensityAnalyzer()
    scores = docs["abstract"].map(lambda t: text_sentiment(t, analyzer))

    out = pd.DataFrame({
        "id": docs["id"].to_numpy(),
        "topic_strict": strict >= 0,  # True = HDBSCAN put it in a topic by itself
        "topic_id": filled,           # outliers got their nearest topic when close enough
        "topic_label": [topic_label(t, words) for t in filled],
        "sentiment_score": scores.round(4).to_numpy(),
        "sentiment": scores.map(label_from_score).to_numpy(),
    })
    out.to_parquet(OUT_PATH, index=False)

    before, after = float((strict == -1).mean()), float((filled == -1).mean())
    counts = out["sentiment"].value_counts()
    report = {
        "outlier_share_before": round(before, 4),
        "outlier_share_after": round(after, 4),
        "outlier_assignment_threshold": round(threshold, 4),
        "sentiment_counts": {k: int(v) for k, v in counts.items()},
        "sentiment_score_mean": round(float(scores.mean()), 4),
        "sentiment_score_median": round(float(scores.median()), 4),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Outliers: {before:.1%} -> {after:.1%} (similarity threshold {threshold:.3f})")
    print("Sentiment counts:", dict(counts))
    print(f"Score mean {scores.mean():.3f}, median {scores.median():.3f}")
    merged = docs.assign(score=scores.to_numpy())
    for name, sub in [("most negative", merged.nsmallest(5, "score")),
                      ("most positive", merged.nlargest(5, "score"))]:
        print(f"\n{name}:")
        for _, r in sub.iterrows():
            print(f"  {r['score']:+.2f}  {r['title'][:90]}")
    print(f"\nSaved {OUT_PATH} and {REPORT_PATH}")


if __name__ == "__main__":
    main()