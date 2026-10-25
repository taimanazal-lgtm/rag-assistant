"""Thin wrapper around an OpenAI-compatible chat API with retries and error handling."""
import os
import time
from dataclasses import dataclass

import openai
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

GROQ_URL = "https://api.groq.com/openai/v1"
RETRYABLE = (
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
)


class LLMError(Exception):
    """The LLM call could not be completed. `kind` says why."""

    def __init__(self, kind, message):
        super().__init__(f"[{kind}] {message}")
        self.kind = kind


@dataclass
class LLMResult:
    text: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float  # includes waiting between retries
    attempts: int
    truncated: bool  # the answer was cut because max_tokens was reached


class LLMClient:
    def __init__(self, model, api_key=None, base_url=GROQ_URL, timeout=30.0,
                 max_retries=3, base_delay=1.0, max_delay=20.0,
                 client=None, sleep=time.sleep):
        self.model = model
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self._sleep = sleep
        if client is None:
            key = api_key or os.getenv("GROQ_API_KEY")
            if not key:
                raise LLMError("auth", "GROQ_API_KEY is missing (check your .env file)")
            client = OpenAI(api_key=key, base_url=base_url, timeout=timeout, max_retries=0)
        self._client = client

    def _delay(self, attempt, error):
        """Seconds to wait: the server's Retry-After if given, else exponential backoff."""
        retry_after = None
        response = getattr(error, "response", None)
        if response is not None:
            try:
                retry_after = float(response.headers.get("retry-after"))
            except (TypeError, ValueError):
                retry_after = None
        delay = retry_after if retry_after is not None else self.base_delay * 2 ** (attempt - 1)
        return min(delay, self.max_delay)

    def chat(self, messages, max_tokens=1500, temperature=0.0):
        start = time.perf_counter()
        attempt = 0
        while True:
            attempt += 1
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                break
            except RETRYABLE as e:
                if attempt > self.max_retries:
                    raise LLMError(
                        "unavailable",
                        f"gave up after {attempt} attempts ({type(e).__name__})",
                    ) from e
                self._sleep(self._delay(attempt, e))
            except openai.AuthenticationError as e:
                raise LLMError("auth", "invalid API key") from e
            except openai.NotFoundError as e:
                raise LLMError(
                    "model_not_found",
                    f"model '{self.model}' not found, run scripts/list_models.py",
                ) from e
            except openai.BadRequestError as e:
                text = str(e).lower()
                too_long = "context" in text or "too long" in text or "reduce the length" in text
                raise LLMError("context_too_long" if too_long else "bad_request", str(e)) from e
            except openai.APIStatusError as e:
                kind = "request_too_large" if e.status_code == 413 else "api_error"
                raise LLMError(kind, f"status {e.status_code}") from e

        choice = resp.choices[0]
        text = (choice.message.content or "").strip()
        finish = choice.finish_reason
        truncated = finish == "length"
        if not text:
            if truncated:
                raise LLMError(
                    "empty_truncated",
                    "the model used all max_tokens before answering, raise max_tokens",
                )
            raise LLMError("empty", "the model returned an empty answer")

        usage = resp.usage
        return LLMResult(
            text=text,
            finish_reason=finish,
            prompt_tokens=getattr(usage, "prompt_tokens", 0),
            completion_tokens=getattr(usage, "completion_tokens", 0),
            latency_s=time.perf_counter() - start,
            attempts=attempt,
            truncated=truncated,
        )