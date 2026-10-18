# RAG Assistant over ArXiv Papers

End-to-end Retrieval-Augmented Generation (RAG) project built during the AI/ML Engineer internship.
The goal is a hallucination-resistant assistant that answers questions grounded in a corpus of ArXiv research papers.

## Progress

| Week | Topic | Status |
|------|-------|--------|
| 1 | Environment, data ingestion & preprocessing | Done |
| 2 | Chunking, embeddings & vector DB | Done |

## Week 1: Environment, Data Ingestion & Preprocessing

### What was built
- Reproducible environment (venv, pinned `requirements.txt`) and an environment check script.
- Data collection script that downloads **8,076** ArXiv papers (title + abstract) from `cs.AI`, `cs.CL`, `cs.CV`, `cs.LG`.
- Modular text-cleaning pipeline (`src/preprocessing.py`): HTML stripping, Unicode normalization, LaTeX cleanup, whitespace cleanup, language filtering.
- 21 unit tests for the cleaning functions.
- Validated clean corpus saved as Parquet.

### Approach
- **Data source:** the official ArXiv API. Abstracts are short and clean, and each paper keeps its ID and URL so answers can cite sources later.
- **Variety:** 4 categories (2,500 papers requested from each, duplicates across categories removed) so topic modeling in later weeks has distinct themes.
- **Cleaning:** the sample data showed LaTeX markup (`\%`, `\emph{}`, `$...$`), so cleaning focuses on that. HTML removal only targets real tags, so math such as `x < y` is not damaged.
- **Validation:** `build_clean_corpus.py` asserts at least 5,000 documents, unique IDs, and no empty text, and counts leftover backslashes. Inspecting the real output (not only the unit tests) found three rounds of LaTeX leftovers (escaped braces, spacing commands like `\,`, math delimiters, `\|`, `\-`). Each was fixed and covered by a new unit test, bringing leftovers from 101 documents down to 0.

### Results
| Metric | Value |
|--------|-------|
| Raw documents | 8,076 |
| Clean documents | 8,076 |
| Abstracts modified by cleaning | 1,925 / 8,076 |
| Words per document (mean) | ~206 |
| Words per document (min / max) | 42 / 349 |
| Unit tests | 21 passed |

Cleaning effect on abstracts (number of abstracts containing each issue):

| Issue | Before | After |
|-------|--------|-------|
| Backslash (LaTeX) | 1,356 | 0 |
| Curly braces | 756 | 0 |
| Newlines / tabs | 257 | 0 |
| Double spaces | 258 | 0 |
| Dollar sign (`$`) | 1,246 | 23 |

The 23 remaining `$` are real currency amounts (e.g. `$5,694`), kept on purpose. Reproduce with `python scripts/compare_before_after.py`.

### Known limitations
- LaTeX math is stripped, so some symbols (e.g. `\geq`, `\leq`) are lost from formulas.
- No papers were removed by the length, duplicate or language filters, because the source is clean English text. The filters are tested through the unit tests.
- Currency symbols (`$`) are preserved, so a few documents with unpaired math `$` may keep a stray dollar sign.

## Week 2: Chunking, Embeddings & Vector DB

### What was built
- Recursive chunking module (`src/chunking.py`) with unit tests.
- Embeddings for all chunks with `sentence-transformers`, with timing logged to `results/embedding_log.json`.
- Persistent ChromaDB collection with metadata (paper id, title, category, URL) and semantic search (`src/vector_store.py`).
- Retrieval benchmark script that logs latency per query type to `results/retrieval_benchmark.json`.
- 21 new unit tests for chunking and the vector store (42 tests in total).

### Chunking strategy
- **Recursive splitting:** tries to cut at the largest natural boundary first (paragraph, line, sentence `". "`, word), and only falls back to a hard character cut if no separator exists. This tries to avoid ending chunks in the middle of a sentence.
- **Size 800 characters, overlap 200 characters.** The embedding model reads at most 256 tokens (about 1,000 characters) and truncates the rest, so chunks are kept below that. Overlap keeps context across chunk borders.
- Each chunk keeps `doc_id`, `chunk_index`, title, categories and URL so answers can cite the source paper.
- The first overlap value (100 characters) produced no shared text between the chunks of the sample document inspected, because scientific sentences are often longer than 100 characters. The overlap was raised to 200, and the share of chunk pairs with real overlap was then measured (67%).

| Chunking result | Value |
|---|---|
| Documents | 8,076 |
| Chunks | 21,967 (mean 2.7 per document) |
| Chunk length (mean / max) | 621 / 800 characters |
| Chunk pairs with real overlap | 9,320 / 13,891 (67%) |
| Chunks shorter than 150 characters | 202 (0.9%) |

### Embeddings
- Model: `all-MiniLM-L6-v2` (384 dimensions), normalized vectors.
- Hardware: NVIDIA GeForce MX350 (2 GB), batch size 64.

| Embedding metric | Value |
|---|---|
| Total time (21,967 chunks) | 187 s |
| Throughput | 117.5 chunks/s (8.5 ms per chunk) |
| Chunks above the 256-token limit | 1 (max 271 tokens, so its end is truncated) |

### ChromaDB
- `PersistentClient` stored in `chroma_db/` (git-ignored), cosine distance.
- Ingestion uses `upsert` in batches of 1,000 (22 s for all chunks), so running it twice never creates duplicates.
- Metadata: `doc_id`, `chunk_index`, `title`, `categories` (stored as one string, since Chroma does not accept lists), `primary_category`, `url`.

**Edge cases handled and tested** (`tests/test_vector_store.py`): re-ingesting the same data, batch ingestion, embeddings/chunks length mismatch, duplicate chunk ids, empty chunk text, search on an empty collection, `k` larger than the collection, metadata filter with no match, and empty or `None` queries.

### Retrieval benchmark
10 queries (keyword, question, long query, filtered, off-topic), each repeated 20 times after a warm-up, on a warm GPU, one query at a time.

| Metric | Result |
|---|---|
| Query embedding | 5.3 to 7.5 ms (median) |
| ChromaDB search (no filter) | 3.1 to 4.1 ms (median) |
| Total, no filter | 8.5 to 11.1 ms (median), p95 up to 14.9 ms |
| Total, with category filter | about 32 to 33 ms (search alone 25 to 27 ms) |
| Search time for k = 1 / 10 / 50 | 3.8 / 3.9 / 5.2 ms (median) |
| Self-retrieval check (200 random chunks) | 200 / 200 found at rank 1 |

Top similarity scores were about 0.5 to 0.7 for on-topic queries and 0.21 to 0.23 for an off-topic one ("chocolate cake"). This may be useful later for refusing out-of-scope questions, but a threshold needs proper evaluation first.

### Known limitations
- One chunk exceeds the model's 256-token limit and is truncated.
- About 33% of consecutive chunk pairs have no overlap, mostly because their last sentence is longer than the overlap.
- Filtered search is about 8 times slower than unfiltered search in this setup. The cause was not investigated.
- Relevance was only inspected by eye on 10 queries (by title). Formal Precision/Recall evaluation comes in a later week.
- Latency was measured on one laptop, one query at a time, so it says nothing about concurrent load.

## Project structure
```
rag-assistant/
├── data/                    # git-ignored
│   ├── raw/                 # arxiv_raw.jsonl
│   └── processed/           # clean_corpus.parquet, chunks.parquet, embeddings.npy
├── chroma_db/               # vector database (git-ignored)
├── results/                 # embedding_log.json, retrieval_benchmark.json
├── scripts/
│   ├── verify_env.py
│   ├── download_data.py
│   ├── build_clean_corpus.py
│   ├── compare_before_after.py
│   ├── build_chunks.py
│   ├── embed_chunks.py
│   ├── build_vector_db.py
│   └── benchmark_retrieval.py
├── src/
│   ├── preprocessing.py     # text cleaning
│   ├── chunking.py          # recursive chunking
│   └── vector_store.py      # ChromaDB ingestion + search
├── tests/
│   ├── test_preprocessing.py
│   ├── test_chunking.py
│   └── test_vector_store.py
├── pytest.ini
├── requirements.txt
└── README.md
```

## Setup

Requires Python 3.12. Tested on Windows (PowerShell) with an NVIDIA MX350 GPU.

```powershell
git clone https://github.com/taimanazal-lgtm/rag-assistant.git
cd rag-assistant
python -m venv venv
venv\Scripts\activate

# install all dependencies (the extra index provides PyTorch with CUDA 12.6)
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
python -m spacy download en_core_web_sm
```

## Usage

```powershell
# 1. Verify the environment
python scripts/verify_env.py

# 2. Download the raw data (about 5 minutes, respects ArXiv rate limits)
python scripts/download_data.py

# 3. Clean and validate the corpus
python -m scripts.build_clean_corpus

# 3b. See what the cleaning changed
python scripts/compare_before_after.py

# 4. Run the unit tests
pytest -v

# 5. Week 2 pipeline
python -m scripts.build_chunks
python -m scripts.embed_chunks
python -m scripts.build_vector_db
python -m scripts.benchmark_retrieval
```

Week 1 output: `data/processed/clean_corpus.parquet` with columns
`id, title, abstract, categories, primary_category, published, url, text, n_words`
(`text` = title + abstract).
Week 2 outputs: `chunks.parquet`, `embeddings.npy`, the `chroma_db/` vector database, and the logs in `results/`.

## Dependencies
pandas, numpy, pyarrow, sentence-transformers, chromadb, spacy, beautifulsoup4, langdetect, arxiv, pytest, python-dotenv, torch.
Exact versions are in `requirements.txt`.

## Author
Taima Nazzal