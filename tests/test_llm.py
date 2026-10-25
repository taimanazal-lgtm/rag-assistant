from types import SimpleNamespace

import httpx
import openai
import pytest

from src.llm import LLMClient, LLMError

REQ = httpx.Request("POST", "http://test")


def make_response(text="hi", finish="stop"):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason=finish)],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )


def status_error(cls, status, message="boom", headers=None):
    response = httpx.Response(status, request=REQ, headers=headers or {})
    return cls(message, response=response, body=None)


class FakeClient:
    """Plays back a list of outcomes: an exception is raised, anything else is returned."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def make_llm(outcomes, **kwargs):
    fake = FakeClient(outcomes)
    sleeps = []
    llm = LLMClient("test-model", client=fake, sleep=sleeps.append, **kwargs)
    return llm, fake, sleeps


MSGS = [{"role": "user", "content": "hello"}]


def test_success_first_try():
    llm, fake, sleeps = make_llm([make_response("answer")])
    result = llm.chat(MSGS)
    assert result.text == "answer"
    assert result.attempts == 1 and fake.calls == 1 and sleeps == []
    assert result.prompt_tokens == 10 and result.completion_tokens == 5


def test_retries_on_rate_limit_then_succeeds():
    llm, fake, sleeps = make_llm([status_error(openai.RateLimitError, 429), make_response()])
    result = llm.chat(MSGS)
    assert result.attempts == 2 and len(sleeps) == 1


def test_respects_retry_after_header():
    err = status_error(openai.RateLimitError, 429, headers={"retry-after": "3"})
    llm, _, sleeps = make_llm([err, make_response()])
    llm.chat(MSGS)
    assert sleeps == [3.0]


def test_retries_on_timeout():
    llm, _, _ = make_llm([openai.APITimeoutError(request=REQ), make_response()])
    assert llm.chat(MSGS).attempts == 2


def test_backoff_doubles_each_attempt():
    errors = [status_error(openai.InternalServerError, 500) for _ in range(3)]
    llm, _, sleeps = make_llm(errors + [make_response()], base_delay=1.0)
    llm.chat(MSGS)
    assert sleeps == [1.0, 2.0, 4.0]


def test_gives_up_after_max_retries():
    errors = [status_error(openai.RateLimitError, 429) for _ in range(3)]
    llm, fake, _ = make_llm(errors, max_retries=2)
    with pytest.raises(LLMError) as exc:
        llm.chat(MSGS)
    assert exc.value.kind == "unavailable"
    assert fake.calls == 3


def test_auth_error_is_not_retried():
    llm, fake, _ = make_llm([status_error(openai.AuthenticationError, 401)])
    with pytest.raises(LLMError) as exc:
        llm.chat(MSGS)
    assert exc.value.kind == "auth" and fake.calls == 1


def test_model_not_found_is_not_retried():
    llm, fake, _ = make_llm([status_error(openai.NotFoundError, 404)])
    with pytest.raises(LLMError) as exc:
        llm.chat(MSGS)
    assert exc.value.kind == "model_not_found" and fake.calls == 1


def test_context_too_long_is_detected():
    err = status_error(openai.BadRequestError, 400, message="maximum context length exceeded")
    llm, _, _ = make_llm([err])
    with pytest.raises(LLMError) as exc:
        llm.chat(MSGS)
    assert exc.value.kind == "context_too_long"


def test_request_too_large_413():
    llm, _, _ = make_llm([status_error(openai.APIStatusError, 413)])
    with pytest.raises(LLMError) as exc:
        llm.chat(MSGS)
    assert exc.value.kind == "request_too_large"


def test_empty_answer_after_hitting_max_tokens():
    llm, _, _ = make_llm([make_response("", "length")])
    with pytest.raises(LLMError) as exc:
        llm.chat(MSGS)
    assert exc.value.kind == "empty_truncated"


def test_truncated_answer_is_flagged():
    llm, _, _ = make_llm([make_response("partial answer", "length")])
    result = llm.chat(MSGS)
    assert result.truncated is True and result.text == "partial answer"


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(LLMError) as exc:
        LLMClient("test-model")
    assert exc.value.kind == "auth"