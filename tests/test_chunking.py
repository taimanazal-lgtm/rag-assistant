import pandas as pd
import pytest

from src.chunking import chunk_documents, chunk_text


def make_text(n=30):
    return ". ".join(f"Sentence number {i}" for i in range(n))


def test_short_text_is_one_chunk():
    assert chunk_text("Short text.", chunk_size=100, overlap=10) == ["Short text."]


def test_empty_text_gives_no_chunks():
    assert chunk_text("   ", chunk_size=100, overlap=10) == []


def test_chunks_respect_max_size():
    chunks = chunk_text(make_text(), chunk_size=100, overlap=30)
    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks)


def test_no_content_is_lost():
    chunks = chunk_text(make_text(), chunk_size=100, overlap=30)
    joined = " ".join(chunks)
    for i in range(30):
        assert f"Sentence number {i}" in joined


def test_consecutive_chunks_overlap():
    chunks = chunk_text(make_text(), chunk_size=100, overlap=30)
    last_sentence = chunks[0].split(". ")[-1]
    assert last_sentence in chunks[1]


def test_long_text_without_separators_is_hard_cut():
    chunks = chunk_text("a" * 250, chunk_size=100, overlap=10)
    assert len(chunks) == 3
    assert all(len(c) <= 100 for c in chunks)


def test_invalid_overlap_raises():
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=50, overlap=50)


def test_chunk_documents_keeps_metadata():
    df = pd.DataFrame([{
        "id": "1234.5678v1",
        "title": "A title",
        "text": make_text(),
        "categories": ["cs.AI", "cs.LG"],
        "url": "http://arxiv.org/abs/1234.5678v1",
    }])
    out = chunk_documents(df, chunk_size=100, overlap=30)
    assert out["chunk_id"].is_unique
    assert out["chunk_id"].iloc[0] == "1234.5678v1_0"
    assert (out["doc_id"] == "1234.5678v1").all()
    assert out["categories"].iloc[0] == "cs.AI, cs.LG"