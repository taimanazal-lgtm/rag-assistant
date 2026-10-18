"""Embed all chunks with sentence-transformers and log the timing."""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = "data/processed/chunks.parquet"
EMB_PATH = "data/processed/embeddings.npy"
LOG_PATH = Path("results/embedding_log.json")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 64


def main():
    chunks = pd.read_parquet(CHUNKS_PATH)
    texts = chunks["text"].tolist()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
    model = SentenceTransformer(MODEL_NAME, device=device)
    print(f"Device: {device_name} | max_seq_length: {model.max_seq_length}")

    # how many chunks are longer than the model limit (they would be truncated)?
    token_lens = [len(ids) for ids in model.tokenizer(texts, truncation=False)["input_ids"]]
    over = sum(n > model.max_seq_length for n in token_lens)
    print(f"Chunks over the token limit: {over}/{len(texts)} | max tokens: {max(token_lens)}")

    model.encode(texts[:BATCH_SIZE], batch_size=BATCH_SIZE)  # warm-up, not timed

    start = time.perf_counter()
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    elapsed = time.perf_counter() - start

    # validation
    assert embeddings.shape[0] == len(chunks), "Row count mismatch"
    assert not np.isnan(embeddings).any(), "NaN in embeddings"

    np.save(EMB_PATH, embeddings)

    log = {
        "model": MODEL_NAME,
        "device": device_name,
        "n_chunks": len(chunks),
        "embedding_dim": int(embeddings.shape[1]),
        "batch_size": BATCH_SIZE,
        "max_seq_length": model.max_seq_length,
        "chunks_over_token_limit": int(over),
        "total_seconds": round(elapsed, 1),
        "chunks_per_second": round(len(chunks) / elapsed, 1),
        "ms_per_chunk": round(1000 * elapsed / len(chunks), 2),
    }
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()