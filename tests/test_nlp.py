import numpy as np
import pandas as pd
import pytest

from src.nlp import (assign_nearest_topic, doc_embeddings, label_from_score,
                     text_sentiment, topic_label)


class FakeAnalyzer:
    def polarity_scores(self, sentence):
        score = 1.0 if "good" in sentence else -1.0 if "bad" in sentence else 0.0
        return {"compound": score}


def unit(degrees):
    r = np.deg2rad(degrees)
    return np.array([np.cos(r), np.sin(r)])


@pytest.mark.parametrize("score, expected", [
    (0.05, "positive"), (0.049, "neutral"), (0.0, "neutral"),
    (-0.049, "neutral"), (-0.05, "negative"),
])
def test_label_from_score_thresholds(score, expected):
    assert label_from_score(score) == expected


def test_text_sentiment_averages_sentences():
    text = "good thing here. good again. bad one."
    assert text_sentiment(text, FakeAnalyzer()) == pytest.approx(1 / 3)


@pytest.mark.parametrize("empty", ["", "   ", None])
def test_text_sentiment_empty_text_is_neutral(empty):
    assert text_sentiment(empty, FakeAnalyzer()) == 0.0


def test_topic_label_unassigned():
    assert topic_label(-1, {}) == "unassigned"


def test_topic_label_uses_first_words():
    assert topic_label(3, {3: "agent, harness, tasks, code"}) == "agent, harness, tasks"


def test_topic_label_missing_topic_falls_back():
    assert topic_label(7, {}) == "topic 7"


def test_assign_nearest_topic_fills_close_outliers_only():
    angles = [-20, -10, 0, 10, 20, 70, 80, 90, 100, 110, 12, 45]
    emb = np.array([unit(a) for a in angles])
    topics = np.array([0] * 5 + [1] * 5 + [-1, -1])
    filled, threshold = assign_nearest_topic(emb, topics)
    assert filled[:10].tolist() == topics[:10].tolist()  # members never change
    assert filled[10] == 0   # close to topic 0, gets it
    assert filled[11] == -1  # halfway between two topics, stays unassigned
    assert 0.9 < threshold < 1.0


def test_assign_nearest_topic_without_outliers_changes_nothing():
    emb = np.array([unit(a) for a in [0, 5, 90, 95]])
    topics = np.array([0, 0, 1, 1])
    filled, _ = assign_nearest_topic(emb, topics)
    assert filled.tolist() == topics.tolist()


def test_doc_embeddings_follow_doc_order_and_are_normalized():
    docs = pd.DataFrame({"id": ["b", "a"]})
    chunks = pd.DataFrame({"doc_id": ["a", "a", "b"]})
    chunk_emb = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 2.0]])
    out = doc_embeddings(docs, chunks, chunk_emb)
    assert np.allclose(out[0], [0.0, 1.0])  # doc "b" first
    assert np.allclose(out[1], [2 ** -0.5, 2 ** -0.5])
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)