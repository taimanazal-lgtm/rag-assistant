import re

import pandas as pd

df = pd.read_parquet("data/processed/clean_corpus.parquet")
pattern = re.compile(r"\\[a-zA-Z]+")

for _, row in df.iterrows():
    found = pattern.findall(row["text"])
    if found:
        print(row["id"], found)
        i = row["text"].find(found[0])
        print("   ...", row["text"][max(0, i - 60): i + 60], "...\n")