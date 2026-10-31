"""Helpers for the NLP metadata: sentiment labels and topic assignment."""
import re

import numpy as np

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
POS_THRESHOLD = 0.05
NEG_THRESHOLD = -0.05


def doc_embeddings(docs, chunks, chunk_emb):
    """Average the chunk embeddings of each document, in the order of `docs`."""
    import pandas as pd

    mean = pd.DataFrame(chunk_emb).groupby(chunks["doc_id"].to_numpy()).mean()
    out = mean.loc[docs["id"]].to_numpy()
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def text_sentiment(text, analyzer):
    """Mean VADER compound score over sentences (a whole abstract saturates near +1 or -1)."""
    sentences = [s for s in SENTENCE_SPLIT.split((text or "").strip()) if s.strip()]
    if not sentences:
        return 0.0
    return float(np.mean([analyzer.polarity_scores(s)["compound"] for s in sentences]))


def label_from_score(score, pos=POS_THRESHOLD, neg=NEG_THRESHOLD):
    if score >= pos:
        return "positive"
    if score <= neg:
        return "negative"
    return "neutral"


def topic_label(topic_id, words_by_topic, n=3):
    """Short readable name from the topic's top words ('unassigned' for -1)."""
    if topic_id < 0:
        return "unassigned"
    words = [w.strip() for w in str(words_by_topic.get(int(topic_id), "")).split(",") if w.strip()]
    return ", ".join(words[:n]) or f"topic {int(topic_id)}"


def assign_nearest_topic(embeddings, topics, quantile=0.10):
    """Give outliers (-1) their nearest topic if they are about as close as typical members.

    The threshold is the `quantile` of how close real members are to their own topic centroid.
    Returns (filled_topics, threshold).
    """
    topics = np.asarray(topics)
    ids = np.array(sorted(t for t in set(topics.tolist()) if t >= 0))
    centroids = np.vstack([embeddings[topics == t].mean(axis=0) for t in ids])
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)

    sims = embeddings @ centroids.T  # cosine similarity (embeddings are normalized)
    best_topic = ids[sims.argmax(axis=1)]
    best_sim = sims.max(axis=1)

    member = topics >= 0
    own_sim = sims[member, np.searchsorted(ids, topics[member])]
    threshold = float(np.quantile(own_sim, quantile))

    filled = topics.copy()
    take = (~member) & (best_sim >= threshold)
    filled[take] = best_topic[take]
    return filled, threshold