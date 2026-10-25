"""Summarize results/rag_log.jsonl: latency and token usage per question."""
import json
from pathlib import Path

lines = Path("results/rag_log.jsonl").read_text(encoding="utf-8").splitlines()
records = [json.loads(line) for line in lines if line.strip()]

print(f"{'status':<19}{'llm_ms':>8}{'total_ms':>10}{'prompt_tok':>12}{'compl_tok':>11}  question")
for r in records:
    t, tok = r["timings_ms"], r["tokens"]
    print(f"{r['status']:<19}{t['llm_ms']:>8.0f}{t['total_ms']:>10.0f}"
          f"{tok.get('prompt', 0):>12}{tok.get('completion', 0):>11}  {r['question'][:45]}")