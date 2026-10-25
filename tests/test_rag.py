import json
import uuid

import chromadb
import numpy as np
import pandas as pd
import pytest

from src.llm import LLMError, LLMResult
from src.rag import (
    OFF_TOPIC_MESSAGE,
    REFUSAL,
    RAGPipeline,
    build_messages,
    check_answer,
    extract_citations,
    split_sentences,
    unsupported_numbers,
)
from src.vector_store import get_collection, ingest

DIM = 8
GOOD_ANSWER = "The model reaches 92 percent accuracy on the benchmark [1]."


def unit(i):
    v = np.zeros(DIM)
    v[i] = 1.0
    return v


def fake_embed(texts):
    """Sentences starting with 'Totally' point away from the sources."""
    vec = unit(1) if texts[0].startswith("Totally") else unit(0)
    return np.tile(vec, (len(texts), 1))


class FakeModel:
    def __init__(self, vec):
        self.vec = vec

    def encode(self, x, **kwargs):
        if isinstance(x, str):
            return self.vec
        return np.tile(self.vec, (len(x), 1))


class FakeLLM:
    def __init__(self, text=None, error=None, truncated=False):
        self.text, self.error, self.truncated, self.calls = text, error, truncated, 0

    def chat(self, messages, max_tokens=1500, temperature=0.0):
        self.calls += 1
        if self.error:
            raise self.error
        return LLMResult(text=self.text, finish_reason="stop", prompt_tokens=100,
                         completion_tokens=20, latency_s=0.01, attempts=1,
                         truncated=self.truncated)


@pytest.fixture
def collection():
    col = get_collection(chromadb.EphemeralClient(), name=f"rag_{uuid.uuid4().hex}")
    chunks = pd.DataFrame([{
        "chunk_id": "p1_0", "doc_id": "p1", "chunk_index": 0,
        "text": "The model reaches 92 percent accuracy on the benchmark.",
        "title": "Paper A", "categories": "cs.AI", "primary_category": "cs.AI",
        "url": "http://x/p1",
    }])
    ingest(col, chunks, np.array([unit(0)]))
    return col


def make_pipeline(collection, llm, query_vec, tmp_path, **kwargs):
    return RAGPipeline(collection, FakeModel(query_vec), llm,
                       log_path=tmp_path / "log.jsonl", **kwargs)


# ---------- prompt ----------
def test_build_messages_has_sources_question_and_rules():
    sources = [{"title": "Paper A", "doc_id": "p1", "text": "some text"},
               {"title": "Paper B", "doc_id": "p2", "text": "other text"}]
    messages = build_messages("What is X?", sources)
    assert messages[0]["role"] == "system" and REFUSAL in messages[0]["content"]
    user = messages[1]["content"]
    assert "[1] Title: Paper A" in user and "[2] Title: Paper B" in user
    assert "Question: What is X?" in user


# ---------- hallucination checks ----------
def test_extract_citations_handles_groups():
    assert extract_citations("A [1] and B [2][3] and C [1, 4].") == {1, 2, 3, 4}


def test_invalid_citation_is_flagged():
    result = check_answer("Some claim about the topic here [5].", ["source text"], fake_embed)
    assert result["invalid_citations"] == [5] and result["flagged"]


def test_missing_citation_is_flagged():
    result = check_answer("A claim without any citation at all.", ["source text"], fake_embed)
    assert "no citations" in result["reasons"]


def test_unsupported_numbers_ignores_citation_markers():
    answer = "It reaches 92 percent and 99 percent [1]."
    assert unsupported_numbers(answer, ["accuracy 92 percent"]) == ["99"]


def test_split_sentences_drops_short_fragments():
    assert split_sentences("Yes. This is a longer sentence here. Ok.") == ["This is a longer sentence here."]


def test_grounded_answer_is_not_flagged():
    result = check_answer(GOOD_ANSWER, ["The model reaches 92 percent accuracy."], fake_embed)
    assert result["flagged"] is False and result["grounding_score"] == 1.0


def test_low_grounding_is_flagged():
    answer = "Totally unrelated claim about something else [1]."
    result = check_answer(answer, ["accuracy 92 percent"], fake_embed)
    assert result["grounding_score"] == 0.0 and result["flagged"]


# ---------- pipeline ----------
def test_answered_question(collection, tmp_path):
    llm = FakeLLM(GOOD_ANSWER)
    r = make_pipeline(collection, llm, unit(0), tmp_path).ask("What accuracy does the model get?")
    assert r.status == "answered" and llm.calls == 1
    assert r.checks["flagged"] is False
    assert len(r.sources) == 1 and r.timings_ms["total_ms"] >= r.timings_ms["embed_ms"]


def test_off_topic_never_calls_llm(collection, tmp_path):
    llm = FakeLLM(GOOD_ANSWER)
    r = make_pipeline(collection, llm, unit(1), tmp_path).ask("best chocolate cake recipe")
    assert r.status == "refused_off_topic" and r.answer == OFF_TOPIC_MESSAGE
    assert llm.calls == 0


def test_model_refusal_with_curly_apostrophe(collection, tmp_path):
    llm = FakeLLM(REFUSAL.replace("'", "\u2019"))
    r = make_pipeline(collection, llm, unit(0), tmp_path).ask("something in scope")
    assert r.status == "refused_by_model"


def test_llm_failure_returns_error_status(collection, tmp_path):
    llm = FakeLLM(error=LLMError("unavailable", "gave up"))
    r = make_pipeline(collection, llm, unit(0), tmp_path).ask("something in scope")
    assert r.status == "error" and "unavailable" in r.error


def test_empty_question_raises(collection, tmp_path):
    with pytest.raises(ValueError):
        make_pipeline(collection, FakeLLM("x"), unit(0), tmp_path).ask("   ")


def test_each_question_is_logged(collection, tmp_path):
    pipe = make_pipeline(collection, FakeLLM(GOOD_ANSWER), unit(0), tmp_path)
    pipe.ask("What accuracy does the model get?")
    lines = (tmp_path / "log.jsonl").read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[0])
    assert len(lines) == 1 and record["status"] == "answered" and "total_ms" in record["timings_ms"]


def test_truncated_answer_is_flagged(collection, tmp_path):
    llm = FakeLLM(GOOD_ANSWER, truncated=True)
    r = make_pipeline(collection, llm, unit(0), tmp_path).ask("What accuracy does the model get?")
    assert r.checks["flagged"] and "answer was cut at max_tokens" in r.checks["reasons"]


def test_empty_collection_is_refused(tmp_path):
    empty = get_collection(chromadb.EphemeralClient(), name=f"empty_{uuid.uuid4().hex}")
    llm = FakeLLM(GOOD_ANSWER)
    r = make_pipeline(empty, llm, unit(0), tmp_path).ask("anything")
    assert r.status == "refused_off_topic" and llm.calls == 0