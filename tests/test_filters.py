import pytest

from src.filters import build_where


def test_no_filters_returns_none():
    assert build_where() is None


def test_topic_only():
    assert build_where(topic=3) == {"topic_id": 3}


def test_topic_zero_is_not_ignored():
    assert build_where(topic=0) == {"topic_id": 0}  # 0 is a valid topic id


def test_topic_string_is_converted():
    assert build_where(topic="5") == {"topic_id": 5}


def test_strict_topic_adds_second_condition():
    where = build_where(topic=2, strict_topic=True)
    assert where == {"$and": [{"topic_id": 2}, {"topic_strict": True}]}


def test_category_and_sentiment_combined():
    where = build_where(category="cs.CV", sentiment="negative")
    assert where == {"$and": [{"primary_category": "cs.CV"}, {"sentiment": "negative"}]}


def test_invalid_sentiment_raises():
    with pytest.raises(ValueError):
        build_where(sentiment="happy")


def test_strict_without_topic_raises():
    with pytest.raises(ValueError):
        build_where(strict_topic=True)