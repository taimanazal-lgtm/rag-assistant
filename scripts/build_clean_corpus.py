"""Clean the raw ArXiv data and save data/processed/clean_corpus.parquet"""
import pandas as pd

from src.preprocessing import clean_text, is_english

RAW_PATH = "data/raw/arxiv_raw.jsonl"
OUT_PATH = "data/processed/clean_corpus.parquet"
MIN_WORDS = 30


def main():
    df = pd.read_json(RAW_PATH, lines=True)
    print(f"Raw documents: {len(df)}")

    # 1. clean text
    df["title"] = df["title"].map(clean_text)
    df["abstract"] = df["abstract"].map(clean_text)
    df["text"] = df["title"] + ". " + df["abstract"]
    df["n_words"] = df["text"].str.split().str.len()

    # 2. filters (we count what each one removes)
    before = len(df)
    df = df[df["n_words"] >= MIN_WORDS]
    print(f"Dropped too short: {before - len(df)}")

    before = len(df)
    df = df.drop_duplicates(subset="id").drop_duplicates(subset="text")
    print(f"Dropped duplicates: {before - len(df)}")

    before = len(df)
    df = df[df["text"].map(is_english)]
    print(f"Dropped non-English: {before - len(df)}")

    df = df.reset_index(drop=True)

    # 3. validation
    assert len(df) >= 5000, f"Only {len(df)} docs, need 5000+"
    assert df["id"].is_unique, "Duplicate ids found"
    assert df["text"].notna().all(), "Null text found"
    assert (df["text"].str.len() > 0).all(), "Empty text found"
    leftover = df["text"].str.contains(r"\\[a-zA-Z]+", regex=True).sum()
    print(f"Docs with leftover LaTeX commands: {leftover}")

    df.to_parquet(OUT_PATH, index=False)
    print(f"\nSaved {len(df)} clean documents to {OUT_PATH}")
    print(df["n_words"].describe())


if __name__ == "__main__":
    main()