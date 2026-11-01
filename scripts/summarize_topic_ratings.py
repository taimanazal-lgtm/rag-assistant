"""Summarize the hand-rated topic coherence."""
import json
from pathlib import Path

import pandas as pd

df = pd.read_csv("labels/topic_coherence.csv")
df["coherence"] = df["coherence"].astype(str).str.strip().str.lower()
order = ["good", "mixed", "bad"]
assert df["coherence"].isin(order).all(), "every topic needs good / mixed / bad"

counts = df["coherence"].value_counts().reindex(order, fill_value=0)
docs = df.groupby("coherence")["size"].sum().reindex(order, fill_value=0)
share_topics = counts["good"] / len(df)
share_docs = docs["good"] / docs.sum()
not_good = [int(t) for t in df.loc[df["coherence"] != "good", "topic_id"]]

print("Topics rated:", len(df))
print(counts.to_string())
print(f"Share of topics rated good: {share_topics:.1%}")
print(f"Share of topic documents inside good topics: {share_docs:.1%}")
print("Not good:", not_good)

Path("results").mkdir(exist_ok=True)
Path("results/topic_ratings.json").write_text(json.dumps({
    "n_topics": len(df),
    "counts": {k: int(v) for k, v in counts.items()},
    "share_topics_good": round(float(share_topics), 4),
    "share_documents_in_good_topics": round(float(share_docs), 4),
    "not_good_topic_ids": not_good,
}, indent=2), encoding="utf-8")