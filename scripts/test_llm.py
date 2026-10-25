"""Smoke test: one call to the LLM API."""
import os
import sys
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

model = sys.argv[1] if len(sys.argv) > 1 else "openai/gpt-oss-20b"
client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
)

start = time.perf_counter()
resp = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "In one short sentence: what is retrieval-augmented generation?"}],
    max_tokens=1000,
    temperature=0,
)
elapsed = time.perf_counter() - start

print(resp.choices[0].message.content)
print(f"\nmodel={resp.model} | {elapsed:.2f}s | usage={resp.usage}")