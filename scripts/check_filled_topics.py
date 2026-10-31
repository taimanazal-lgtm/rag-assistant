"""Rough check: do nearest-topic assignments agree with arXiv categories like strict ones do?"""
import pandas as pd

docs = pd.read_parquet("data/processed/clean_corpus.parquet")[["id", "primary_category"]]
nlp = pd.read_parquet("data/processed/doc_nlp.parquet")
df = docs.merge(nlp, on="id")
df = df[df["topic_id"] >= 0]

strict = df[df["topic_strict"]]
dominant = strict.groupby("topic_id")["primary_category"].agg(lambda s: s.value_counts().index[0])
df["same_category_as_topic"] = df["primary_category"] == df["topic_id"].map(dominant)

summary = df.groupby("topic_strict")["same_category_as_topic"].agg(["mean", "size"])
summary.index = summary.index.map({True: "strict (HDBSCAN)", False: "filled (nearest topic)"})
print(summary.round(3))