import uuid

import chromadb
import numpy as np
import pandas as pd
import pytest

from src.vector_store import get_collection, ingest, search, search_by_embedding


def make_data(n=10, dim=8):
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(n, dim))
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    chunks = pd.DataFrame({
        "chunk_id": [f"d{i}_0" for i in range(n)],
        "doc_id": [f"d{i}" for i in range(n)],
        "chunk_index": [0] * n,
        "text": [f"text {i}" for i in range(n)],
        "title": [f"title {i}" for i in range(n)],
        "categories": ["cs.AI"] * n,
        "primary_category": ["cs.AI" if i % 2 == 0 else "cs.CL" for i in range(n)],
        "url": [f"http://x/{i}" for i in range(n)],
    })
    return chunks, emb


@pytest.fixture
def collection():
    client = chromadb.EphemeralClient()
    return get_collection(client, name=f"test_{uuid.uuid4().hex}")


def test_ingest_count(collection):
    chunks, emb = make_data()
    ingest(collection, chunks, emb)
    assert collection.count() == 10


def test_ingest_twice_does_not_duplicate(collection):
    chunks, emb = make_data()
    ingest(collection, chunks, emb)
    ingest(collection, chunks, emb)
    assert collection.count() == 10


def test_ingest_in_small_batches(collection):
    chunks, emb = make_data()
    ingest(collection, chunks, emb, batch_size=3)
    assert collection.count() == 10


def test_ingest_length_mismatch_raises(collection):
    chunks, emb = make_data()
    with pytest.raises(ValueError):
        ingest(collection, chunks, emb[:5])


def test_ingest_duplicate_ids_raises(collection):
    chunks, emb = make_data()
    chunks.loc[1, "chunk_id"] = chunks.loc[0, "chunk_id"]
    with pytest.raises(ValueError):
        ingest(collection, chunks, emb)


def test_search_returns_closest_first(collection):
    chunks, emb = make_data()
    ingest(collection, chunks, emb)
    results = search_by_embedding(collection, emb[3], k=3)
    assert results[0]["chunk_id"] == "d3_0"
    assert results[0]["distance"] < 1e-4
    assert results[0]["distance"] <= results[1]["distance"] <= results[2]["distance"]


def test_search_k_larger_than_collection(collection):
    chunks, emb = make_data()
    ingest(collection, chunks, emb)
    assert len(search_by_embedding(collection, emb[0], k=100)) == 10


def test_search_empty_collection(collection):
    assert search_by_embedding(collection, np.ones(8), k=5) == []


def test_search_with_metadata_filter(collection):
    chunks, emb = make_data()
    ingest(collection, chunks, emb)
    results = search_by_embedding(collection, emb[0], k=10, where={"primary_category": "cs.CL"})
    assert len(results) == 5
    assert all(r["primary_category"] == "cs.CL" for r in results)


def test_search_filter_with_no_match(collection):
    chunks, emb = make_data()
    ingest(collection, chunks, emb)
    assert search_by_embedding(collection, emb[0], k=5, where={"primary_category": "cs.RO"}) == []


@pytest.mark.parametrize("bad_query", ["", "   ", None])
def test_empty_query_raises(collection, bad_query):
    with pytest.raises(ValueError):
        search(collection, model=None, query=bad_query)