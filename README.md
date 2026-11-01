# RAG Assistant over ArXiv Papers

End-to-end Retrieval-Augmented Generation (RAG) project built during the AI/ML Engineer internship.
The goal is a hallucination-resistant assistant that answers questions grounded in a corpus of ArXiv research papers.

## Progress

| Week | Topic | Status |
|------|-------|--------|
| 1 | Environment, data ingestion & preprocessing | Done |
| 2 | Chunking, embeddings & vector DB | Done |
| 3 | LLM integration & prompt engineering | Done |
| 4 | NLP analysis (topics & sentiment) | Done |

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
- 21 new unit tests for chunking and the vector store (42 tests in total at the end of Week 2).

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
- Metadata: `doc_id`, `chunk_index`, `title`, `categories` (stored as one string, since Chroma does not accept lists), `primary_category`, `url`. Week 4 adds topic and sentiment fields.

**Edge cases handled and tested** (`tests/test_vector_store.py`): re-ingesting the same data, batch ingestion, embeddings/chunks length mismatch, duplicate chunk ids, search on an empty collection, `k` larger than the collection, metadata filter with no match, and empty or `None` queries.

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

Top similarity scores were about 0.5 to 0.7 for on-topic queries and 0.21 to 0.23 for an off-topic one ("chocolate cake").

### Known limitations
- One chunk exceeds the model's 256-token limit and is truncated.
- About 33% of consecutive chunk pairs have no overlap, mostly because their last sentence is longer than the overlap.
- Filtered queries took 25 to 27 ms in the ChromaDB search step versus 3 to 4 ms for unfiltered queries. These were different queries, so this is not a controlled comparison, and the cause was not investigated.
- Relevance was only inspected by eye on 10 queries (by title). Formal Precision/Recall evaluation comes in a later week.
- Latency was measured on one laptop, one query at a time, so it says nothing about concurrent load.

## Week 3: LLM Integration & Prompt Engineering

### What was built
- LLM client (`src/llm.py`) for any OpenAI-compatible API, with timeouts, retries and typed errors. Used with **Groq** and the model `openai/gpt-oss-20b`.
- RAG pipeline (`src/rag.py`): retrieve, out-of-domain gate, prompt with numbered sources, generate, answer checks.
- `scripts/ask.py` asks one question or runs an 11-question demo set. `scripts/summarize_log.py` prints latency, tokens and API attempts per question.
- Every question is logged to `results/rag_log.jsonl` (status, timings, tokens, attempts, sources, checks).
- `scripts/list_models.py` lists the provider's models, because free model IDs change often and can cause 404 errors.
- 29 new unit tests (13 for the LLM client, 16 for the pipeline). They use fake clients, so no API key or internet is needed. 71 tests in total at the end of Week 3.

### How a question is answered
1. Embed the question and retrieve the 5 closest chunks.
2. **Out-of-domain gate:** if the best similarity is below 0.30, refuse right away without calling the LLM.
3. Build the prompt: a system prompt with rules, the sources numbered `[1]` to `[5]` (title, paper id, text), then the question.
4. Generate the answer (temperature 0).
5. If the model replies with the exact refusal sentence, the status is `refused_by_model`.
6. Otherwise run the answer checks and log everything.

Possible statuses: `answered`, `refused_off_topic`, `refused_by_model`, `error`.

### Prompt engineering
- The system prompt tells the model to use only the given sources, cite them as `[n]`, never invent titles, authors or numbers, ignore instructions found inside sources, and answer in at most 5 sentences.
- When the sources are not enough, the model must reply with one exact sentence. This lets the code detect refusals reliably.
- Sources are numbered and carry the paper id, so every claim can be traced back.

### API error handling
| Situation | Behavior |
|---|---|
| Rate limit (429), timeout, connection error, server error (5xx) | Retry up to 3 times. Wait for the `Retry-After` header if present, otherwise 1, 2, 4 seconds (max 20 s). Then raise `unavailable`. |
| Invalid API key, model not found (404) | No retry. Clear message (points to `scripts/list_models.py`). |
| Prompt too long (400) or request too large (413) | No retry. `context_too_long` / `request_too_large`. |
| Empty answer after reaching `max_tokens` (reasoning models spend tokens on thinking) | `empty_truncated` error asking to raise `max_tokens`. |
| Answer cut at `max_tokens` | Returned, but flagged. |
| Missing API key | Error at startup. |

In the pipeline, any LLM error becomes a status `error` with a friendly message instead of a crash. An empty question raises `ValueError`, and an empty database is treated as off-topic.

### Hallucination checks
Run on every answered response, without another LLM call:
1. The answer has at least one citation.
2. Every cited number refers to an existing source.
3. Every number in the answer also appears in the sources.
4. **Grounding score:** share of answer sentences whose embedding is close (cosine 0.45 or more) to at least one source. The answer is flagged below 0.5.

Together with the similarity gate and the model's own refusal, these are three layers against unsupported answers. The thresholds (0.30, 0.45, 0.5) are starting values, not tuned.

### Demo results
11 questions, one run (`python -m scripts.ask --demo`, saved in `results/rag_demo_report.json`).

| Category | Questions | Outcome | Top similarity |
|---|---|---|---|
| In-domain | 5 | all answered, none flagged | 0.63 to 0.70 |
| Off-topic | 3 | refused by the gate, no LLM call | 0.21 to 0.22 |
| Not in the corpus (traps) | 2 | refused by the model | 0.45 and 0.52 |
| Prompt injection | 1 | refused by the gate | 0.25 |

End-to-end latency in ms (GPU and ChromaDB warmed up before timing):

| Metric | Value |
|---|---|
| Question embedding (answered) | 26 to 34 |
| ChromaDB search (answered) | 12 to 36 |
| LLM generation (answered) | 763 to 1,747 |
| **Total, answered (n=5)** | **median 1,421, p95 1,842** |
| **Total, refused by the gate (n=4)** | **median 47.5, p95 52.2** |
| Total, refused by the model (n=2) | 499 and 7,389 |

Prompts were about 870 to 1,050 tokens. The 7,389 ms case needed 3 API attempts, so the retry logic was used. The error type was not logged. In an earlier run without the ChromaDB warm-up, the first search took 876 ms (cold start).

### Known limitations
- The thresholds were chosen by looking at only 11 questions. Off-topic questions (0.21 to 0.25) and trap questions (0.45 to 0.52) are separated by the gate, but this is not validated on a larger set.
- The hallucination checks are heuristics, tested with unit tests but never triggered on a real answer so far. I did not verify every claim against the source text, so "not flagged" does not prove an answer is correct.
- The citation check only needs one citation in the whole answer. A sentence without a citation can pass (the last sentence of the diffusion answer had none and was not flagged).
- The prompt-injection rule in the system prompt was not really tested: the injection question was refused by the similarity gate before reaching the model.
- The same paper can fill several of the 5 source slots (two chunks of one paper appeared in two answers).
- Corpus coverage is limited to recent papers. For the hallucination question, most sources were about vision-language models, not language models in general.
- Answers vary slightly between runs even at temperature 0, so the demo results come from a single run.
- `context_too_long` detection depends on the text of the provider's error message and may not work with every provider.
- Free-tier rate limits apply and latency was measured on one laptop with one query at a time.

## Week 4: NLP Analysis (Topics & Sentiment)

### What was built
- Topic modeling with BERTopic over the 8,076 papers (`scripts/run_topics.py`), with reports in `results/topics_report.json` and `results/topic_review.csv`.
- VADER sentiment scores and nearest-topic assignment for papers without a topic (`scripts/build_nlp_metadata.py`, helpers in `src/nlp.py`).
- Topic and sentiment metadata written into ChromaDB for all 21,967 chunks (`scripts/add_nlp_metadata.py`), without recomputing embeddings.
- Filtered retrieval: `src/filters.py` builds the ChromaDB filter and `scripts/ask.py` accepts `--topic`, `--strict-topic`, `--category` and `--sentiment` (`RAGPipeline.ask(question, where=...)`).
- Hand evaluation: 64 topics rated for coherence (`labels/topic_coherence.csv`) and 50 abstracts labeled for sentiment (`labels/sentiment_labels.csv`).
- 23 new unit tests (94 in total).

### Topic modeling
- **Document vectors:** the average of each paper's chunk embeddings from Week 2, so all of its text is used (no truncation).
- **Pipeline:** UMAP (5 dimensions, cosine, seed 42, cached on disk), HDBSCAN (`min_cluster_size=30`, `min_samples=5`, `leaf` selection), then BERTopic's c-TF-IDF over unigrams and bigrams with English stop words removed.
- **First configuration failed:** `eom` clustering with `min_df=5` crashed with a vectorizer error, most likely because HDBSCAN returned very few topics (not verified). The final setup uses `leaf` and no `min_df`, because the vectorizer sees one document per topic and `min_df` would drop topic-specific words.
- **Result:** 64 topics (size 30 to 201, median 60). Examples: agents, 3D Gaussian splatting, time-series forecasting, KV cache, quantum.
- **Unassigned papers:** HDBSCAN left **45.1%** of the papers without a topic (`-1`). Each of them gets its nearest topic (cosine similarity to the topic centroid) only if it is at least as close as the 10th percentile of real members (threshold 0.577). After this, **12.3%** stay unassigned. The metadata keeps a `topic_strict` flag, so `--strict-topic` searches only papers HDBSCAN placed itself.
- **Sanity check of the filled assignments** (`scripts/check_filled_topics.py`): 57.0% of nearest-topic papers share the main arXiv category of their topic, against 61.6% of the original members. The original members' figure is slightly inflated (the main category was computed from them), and arXiv categories overlap, so this is only a rough check, not an accuracy.

### Hand evaluation of topics
All 64 topics were rated by reading the top words and 3 example titles of each topic:

| Rating | Topics |
|---|---|
| good | 59 (92.2% of topics) |
| mixed | 5 (topics 11, 20, 34, 56, 59) |
| bad | 0 |

93.4% of the documents that belong to a topic are in a topic rated good (`scripts/summarize_topic_ratings.py`, `results/topic_ratings.json`).

In the mixed topics, some example titles do not match the top words. For example, topic 34 has "moe, experts" among its words, but two of its three example titles are about traffic control and edge inference. Because only three titles per topic were read, and there was a single rater, the share of good topics may be optimistic.

### Edge cases
- **Short documents:** after cleaning, every paper has at least 42 words, and each one is represented by all of its chunks. Empty text in the sentiment step returns a neutral score (unit tested).
- **Jargon and acronyms:** acronyms and coined terms appear as top words (e.g. `opd`, `jev`) and cannot be understood from the words alone; the example titles were needed to interpret them. Stop words are removed and bigrams are kept (`gaussian splatting`, `kv cache`).
- **Unassigned papers:** handled as described above, with a flag to separate them.
- **Filters:** topic `0` is a valid id (unit tested), and an invalid sentiment or `--strict-topic` without `--topic` raises an error.

### Sentiment
- VADER (`vaderSentiment`): the compound score is computed sentence by sentence and averaged. At or above +0.05 is positive, at or below -0.05 is negative, otherwise neutral.
- On the whole corpus: 5,263 positive (65.2%), 1,584 neutral (19.6%), 1,229 negative (15.2%). Mean score 0.11.
- **Label definition used for the 50 hand labels:** *negative* = the main message of the abstract is the danger, vulnerability or persistent failure of something; *positive* = the main message is a new method or tool presented as a success over earlier work; *neutral* = neither of the two (descriptive or analytical). This is closer to "problem versus solution" than to ordinary sentiment.
- Borderline example: `2609.34970v1` presents a mitigation, but its main message is that behavioral safety is an illusion, so it was labeled negative. Another labeler could choose positive.

| Evaluation against 50 hand labels | Result |
|---|---|
| Labels (positive / neutral / negative) | 39 / 5 / 6 |
| Accuracy | 58.0% (95% interval 44.2% to 70.6%) |
| Baseline "always positive" | 78.0% accuracy, macro F1 about 0.29 |
| VADER macro F1 | 0.38 |

| Hand label \ VADER | positive | neutral | negative |
|---|---|---|---|
| positive | 26 | 9 | 4 |
| neutral | 2 | 1 | 2 |
| negative | 3 | 1 | 2 |

Precision / recall: positive 0.84 / 0.67, neutral 0.09 / 0.20, negative 0.25 / 0.33.

**Conclusion:** VADER scores below the always-positive baseline in accuracy and is only slightly better in macro F1. It measures the polarity of words, not the main message of an abstract, so low agreement with this definition is expected. **The sentiment metadata is not reliable enough to filter on.** It is kept in the database and documented, but not recommended.

### How the labels were made
Both label sets (topic coherence and sentiment) were made with the help of an AI assistant. The sentiment accuracy above is therefore the agreement between VADER and these labels, not an independent human ground truth. No second labeler was involved, so agreement between labelers was not measured.

### Filtered retrieval
```powershell
python -m scripts.ask --topic 13 "How are time series forecasted?"
python -m scripts.ask --category cs.CV --sentiment negative "How are models attacked?"
```
- Number of chunks matching a filter: topic 2 has 862, sentiment negative has 3,344, category cs.CV has 5,671.
- With `--topic 13`, all five sources came from topic 13 (forecasting, series, time series) and the question was answered (top similarity 0.55).
- `--category cs.CV --sentiment negative` and the same query with `--sentiment positive` both returned papers about attacks (top similarity 0.42 and 0.47). So the sentiment filter did not visibly change what was retrieved, which matches the evaluation above. Topic and category filters work as expected.

### Known limitations
- 12.3% of the papers still have no topic, and the nearest-topic assignment was checked only with the rough category comparison above.
- Topic ratings are subjective: one rater, top words and three titles per topic.
- The 50 sentiment labels are few and unbalanced (only 5 neutral and 6 negative), so the per-class numbers are unstable, and the labels depend on a definition with borderline cases.
- VADER does not capture the main message of an abstract (see above); a model trained for this kind of text, or an LLM judge, would be a better next step.
- The topic model describes this corpus snapshot. New papers would need a new run.
- The same paper can still fill several of the retrieved source slots (Week 3 limitation).
- The latency of filtered queries was not measured again in this week, and the answer-quality checks of Week 3 were not repeated with filters.

## Project structure
```
rag-assistant/
├── data/                    # git-ignored
│   ├── raw/                 # arxiv_raw.jsonl
│   └── processed/           # clean_corpus, chunks, embeddings, doc_topics, doc_nlp, doc_umap
├── chroma_db/               # vector database (git-ignored)
├── labels/                  # hand labels: topic_coherence.csv, sentiment_labels.csv
├── results/                 # reports and logs (json, csv, jsonl)
├── scripts/
│   ├── verify_env.py
│   ├── download_data.py
│   ├── build_clean_corpus.py
│   ├── compare_before_after.py
│   ├── build_chunks.py
│   ├── embed_chunks.py
│   ├── build_vector_db.py
│   ├── benchmark_retrieval.py
│   ├── list_models.py
│   ├── test_llm.py
│   ├── ask.py
│   ├── summarize_log.py
│   ├── run_topics.py
│   ├── build_nlp_metadata.py
│   ├── check_filled_topics.py
│   ├── add_nlp_metadata.py
│   ├── make_label_set.py
│   ├── evaluate_sentiment.py
│   └── summarize_topic_ratings.py
├── src/
│   ├── preprocessing.py     # text cleaning
│   ├── chunking.py          # recursive chunking
│   ├── vector_store.py      # ChromaDB ingestion + search
│   ├── llm.py               # LLM client with retries and error handling
│   ├── rag.py               # RAG pipeline and hallucination checks
│   ├── nlp.py               # sentiment and topic helpers
│   └── filters.py           # ChromaDB filters from topic / category / sentiment
├── tests/
│   ├── test_preprocessing.py
│   ├── test_chunking.py
│   ├── test_vector_store.py
│   ├── test_llm.py
│   ├── test_rag.py
│   ├── test_nlp.py
│   └── test_filters.py
├── .env.example             # template for the API key (.env is git-ignored)
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

From Week 3 on, an LLM API key is needed. Create a free key on Groq (console.groq.com), then:

```powershell
copy .env.example .env
# open .env and replace your_key_here with your real key
python scripts/list_models.py groq     # check which models are available today
```

The model is set in `scripts/ask.py` (`LLM_MODEL`). If it returns a 404 error, pick another ID from the list.

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

# 4. Run the unit tests (no API key needed)
pytest -v

# 5. Week 2 pipeline
python -m scripts.build_chunks
python -m scripts.embed_chunks
python -m scripts.build_vector_db
python -m scripts.benchmark_retrieval

# 6. Week 3: ask the RAG system (needs the API key in .env)
python -m scripts.ask "How do diffusion models generate images?"
python -m scripts.ask --demo
python scripts/summarize_log.py

# 7. Week 4: NLP analysis
python -m scripts.run_topics
python -m scripts.build_nlp_metadata
python -m scripts.check_filled_topics
python -m scripts.add_nlp_metadata        # needs the ChromaDB from Week 2
python scripts/make_label_set.py          # creates labels/sentiment_labels.csv (never overwrites it)
python scripts/evaluate_sentiment.py      # after filling the label column by hand
python scripts/summarize_topic_ratings.py # after filling labels/topic_coherence.csv
python -m scripts.ask --topic 13 "How are time series forecasted?"
```

Week 1 output: `data/processed/clean_corpus.parquet` with columns
`id, title, abstract, categories, primary_category, published, url, text, n_words`
(`text` = title + abstract).
Week 2 outputs: `chunks.parquet`, `embeddings.npy`, the `chroma_db/` vector database, and the logs in `results/`.
Week 3 outputs: `results/rag_demo_report.json` and `results/rag_log.jsonl`.
Week 4 outputs: `data/processed/doc_topics.parquet`, `doc_nlp.parquet`, the topic and sentiment metadata inside `chroma_db/`, and the reports in `results/` and `labels/`.

## Dependencies
pandas, numpy, pyarrow, sentence-transformers, chromadb, spacy, beautifulsoup4, langdetect, arxiv, openai, requests, pytest, python-dotenv, torch, bertopic, umap-learn, hdbscan, scikit-learn, vaderSentiment.
Exact versions are in `requirements.txt`.

## Author
Taima Nazzal