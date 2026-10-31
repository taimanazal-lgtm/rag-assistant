"""Create a random sample of abstracts to label by hand (sentiment ground truth)."""
import sys
from pathlib import Path

import pandas as pd

OUT = Path("labels/sentiment_labels.csv")
N = 50
SEED = 42

if OUT.exists():
    sys.exit(f"{OUT} already exists. Not overwriting your labels.")

docs = pd.read_parquet("data/processed/clean_corpus.parquet")
sample = docs.sample(N, random_state=SEED)[["id", "abstract"]].copy()
sample["label"] = ""  # positive / neutral / negative
OUT.parent.mkdir(exist_ok=True)
sample.to_csv(OUT, index=False, encoding="utf-8-sig")
print(f"Saved {N} abstracts to {OUT}. Fill the 'label' column by hand.")