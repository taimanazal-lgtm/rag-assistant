# RAG Assistant over ArXiv Papers

End-to-end Retrieval-Augmented Generation (RAG) project built during the AI/ML Engineer internship.
The goal is a hallucination-resistant assistant that answers questions grounded in a corpus of ArXiv research papers.

## Progress

| Week | Topic | Status |
|------|-------|--------|
| 1 | Environment, data ingestion & preprocessing | Done |

## Week 1: Environment, Data Ingestion & Preprocessing

### What was built
- Reproducible environment (venv, pinned `requirements.txt`) and an environment check script.
- Data collection script that downloads **8,076** ArXiv papers (title + abstract) from `cs.AI`, `cs.CL`, `cs.CV`, `cs.LG`.
- Modular text-cleaning pipeline (`src/preprocessing.py`): HTML stripping, Unicode normalization, LaTeX cleanup, whitespace cleanup, language filtering.
- 13 unit tests for the cleaning functions.
- Validated clean corpus saved as Parquet.

### Approach
- **Data source:** the official ArXiv API. Abstracts are short and clean, and each paper keeps its ID and URL so answers can cite sources later.
- **Variety:** 4 categories (2,500 papers requested from each, duplicates across categories removed) so topic modeling in later weeks has distinct themes.
- **Cleaning:** the sample data showed LaTeX markup (`\%`, `\emph{}`, `$...$`), so cleaning focuses on that. HTML removal only targets real tags, so math such as `x < y` is not damaged.
- **Validation:** `build_clean_corpus.py` asserts at least 5,000 documents, unique IDs, and no empty text. Inspecting the output found a bug (escaped braces `\{ \}` left stray backslashes in 12 documents). It was fixed and covered by a new test.

### Results
| Metric | Value |
|--------|-------|
| Raw documents | 8,076 |
| Clean documents | 8,076 |
| Words per document (mean) | ~206 |
| Words per document (min / max) | 42 / 349 |
| Unit tests | 13 passed |
| Documents with leftover LaTeX commands | 0 |

### Known limitations
- LaTeX math is stripped, so some symbols (e.g. `\geq`, `\leq`) are lost from formulas.
- Some words appear glued together (e.g. `timeto`). They already exist in the downloaded data, so they are left as is rather than risking damage to valid words.
- No papers were removed by the length, duplicate or language filters, because the source is clean English text. The filters are tested through the unit tests.

## Project structure
```
rag-assistant/
├── data/
│   ├── raw/                 # downloaded data (git-ignored)
│   └── processed/           # clean_corpus.parquet (git-ignored)
├── scripts/
│   ├── verify_env.py        # environment check (GPU, imports, spaCy, ChromaDB)
│   ├── download_data.py     # downloads ArXiv papers
│   ├── build_clean_corpus.py# cleans + validates + saves the corpus
│   └── inspect_leftovers.py # debugging helper for leftover LaTeX
├── src/
│   └── preprocessing.py     # cleaning functions
├── tests/
│   └── test_preprocessing.py
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

# 4. Run the unit tests
pytest -v
```

Output: `data/processed/clean_corpus.parquet` with columns
`id, title, abstract, categories, primary_category, published, url, text, n_words`
(`text` = title + abstract, used for embeddings in the next weeks).

## Dependencies
pandas, pyarrow, sentence-transformers, chromadb, spacy, beautifulsoup4, langdetect, arxiv, pytest, python-dotenv, torch.
Exact versions are in `requirements.txt`.

## Author
Taima Nazzal