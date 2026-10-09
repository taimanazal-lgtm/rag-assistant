import pandas as pd

raw = pd.read_json("data/raw/arxiv_raw.jsonl", lines=True)
clean = pd.read_parquet("data/processed/clean_corpus.parquet")

df = raw.merge(clean[["id", "abstract"]], on="id", suffixes=("_raw", "_clean"))

checks = {
    "backslash (LaTeX)": r"\\",
    "dollar sign ($)": r"\$",
    "curly braces": r"[{}]",
    "newlines/tabs": r"[\n\t]",
    "double spaces": r"  ",
}

print(f"{'issue':<20}{'raw':>8}{'clean':>8}")
for name, pattern in checks.items():
    r = df["abstract_raw"].str.contains(pattern, regex=True).sum()
    c = df["abstract_clean"].str.contains(pattern, regex=True).sum()
    print(f"{name:<20}{r:>8}{c:>8}")

changed = (df["abstract_raw"] != df["abstract_clean"]).sum()
print(f"\nAbstracts modified by cleaning: {changed} / {len(df)}")