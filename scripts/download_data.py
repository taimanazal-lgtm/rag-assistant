"""Download ArXiv abstracts into data/raw/arxiv_raw.jsonl"""
import json
from pathlib import Path

import arxiv

CATEGORIES = ["cs.AI", "cs.CL", "cs.CV", "cs.LG"]
PER_CATEGORY = 2500
OUT_PATH = Path("data/raw/arxiv_raw.jsonl")


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    client = arxiv.Client(page_size=100, delay_seconds=3, num_retries=5)
    seen = set()

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for cat in CATEGORIES:
            search = arxiv.Search(
                query=f"cat:{cat}",
                max_results=PER_CATEGORY,
                sort_by=arxiv.SortCriterion.SubmittedDate,
            )
            for paper in client.results(search):
                paper_id = paper.get_short_id()
                if paper_id in seen:
                    continue
                seen.add(paper_id)
                record = {
                    "id": paper_id,
                    "title": paper.title,
                    "abstract": paper.summary,
                    "categories": paper.categories,
                    "primary_category": paper.primary_category,
                    "published": paper.published.isoformat(),
                    "url": paper.entry_id,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"{cat} done | unique so far: {len(seen)}")

    print(f"Saved {len(seen)} papers to {OUT_PATH}")


if __name__ == "__main__":
    main()