# RAG Knowledge Assistant with Retrieval Evaluation

Document Q&A system with a production-grade evaluation pipeline. Upload any document, ask questions, get grounded answers with source citations — and every response is scored against a fixed golden evaluation set before it leaves the system.

---

## What it does

- Indexes uploaded documents with sentence-level chunking
- Retrieves relevant chunks using semantic vector search (ChromaDB + FAISS)
- Returns grounded answers with source passage citations
- Scores every response on three dimensions: relevance, source coverage, and unsupported-answer risk
- Flags answers that make claims not supported by retrieved context

The evaluation layer is not a demo feature. It runs on every query and outputs a structured score object alongside the answer. Answers that exceed unsupported-answer risk threshold are flagged before they reach the user.

---

## Architecture

```
Document Upload
      │
      ▼
[Chunker]  — sentence-level splitting, ~300 token target
      │
      ▼
[Embedder]  — Sentence Transformers (all-MiniLM-L6-v2)
      │
      ▼
[Vector Store]  — ChromaDB (persistent) + FAISS (in-memory for speed)
      │
      ▼
User Query ──► [Retriever]  — top-k semantic search
                    │
                    ▼
               [Generator]  — OpenAI GPT-4o with retrieved context
                    │
                    ▼
               [Evaluator]  — scores: relevance / coverage / unsupported risk
                    │
                    ▼
              Answer + Score + Source Citations
```

---

## Evaluation pipeline

### Golden evaluation set

`evaluation/golden_set.json` contains 25 hand-labelled question-answer pairs covering:
- Direct retrieval (exact keyword match)
- Inference (requires combining information across chunks)
- Multi-section reasoning
- Edge cases (out-of-scope questions, ambiguous references)
- Negation and conditional logic

### Scoring dimensions

| Dimension | What it measures | Flag threshold |
|---|---|---|
| Relevance | Does the answer address the question? | < 0.6 |
| Source coverage | Are claims backed by retrieved chunks? | < 0.7 |
| Unsupported answer risk | Does the answer contain claims outside retrieved context? | > 0.3 |

### Running evaluation

```bash
python evaluation/run_eval.py --golden evaluation/golden_set.json --output evaluation/results/
```

Output: `evaluation/results/eval_YYYY-MM-DD.json` with per-question scores and aggregate metrics.

---

## Tech stack

| Layer | Technology |
|---|---|
| Chunking | Custom sentence-level splitter |
| Embeddings | Sentence Transformers (all-MiniLM-L6-v2) |
| Vector store | ChromaDB (persistent) + FAISS (fast retrieval) |
| LLM | OpenAI API (GPT-4o) |
| Evaluation | Custom scoring against golden_set.json |
| UI | Streamlit |

---

## Quickstart

### Prerequisites
- Python 3.11+
- OpenAI API key

### Run locally

```bash
git clone https://github.com/Avvv19/rag-search-answer-evaluation-assistant
cd rag-search-answer-evaluation-assistant
cp .env.example .env          # add OPENAI_API_KEY
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501. Upload a PDF or text file, then ask questions.

---

## Environment variables

```
OPENAI_API_KEY=sk-...
CHROMA_PERSIST_DIR=./chroma_db
EVAL_THRESHOLD_RELEVANCE=0.6
EVAL_THRESHOLD_COVERAGE=0.7
EVAL_THRESHOLD_UNSUPPORTED=0.3
```

---

## Project structure

```
rag-search-answer-evaluation-assistant/
├── app.py                    # Streamlit entry point
├── rag/
│   ├── chunker.py            # Sentence-level document chunking
│   ├── embedder.py           # Sentence Transformer embedding wrapper
│   ├── retriever.py          # ChromaDB + FAISS retrieval
│   └── generator.py          # OpenAI call with context injection
├── evaluation/
│   ├── golden_set.json       # 25 hand-labelled Q&A pairs
│   ├── scorer.py             # Relevance, coverage, unsupported-risk scoring
│   ├── run_eval.py           # CLI evaluation runner
│   └── results/              # Stored eval results by date
├── tests/
│   ├── test_chunker.py
│   ├── test_retriever.py
│   └── test_scorer.py
├── requirements.txt
└── .env.example
```

---

## Understanding the scores

**Relevance** — cosine similarity between the query embedding and the answer embedding. Measures whether the answer is topically on-target.

**Source coverage** — fraction of answer sentences that have a retrieved chunk with similarity above threshold. Low coverage means the answer is going beyond what was retrieved.

**Unsupported answer risk** — inverse of source coverage, weighted by answer length. A short answer with one unsupported sentence scores differently from a long answer with one unsupported sentence.

When all three dimensions are within threshold: the answer is presented normally.
When any dimension fails: the answer is presented with a yellow warning flag and the specific dimension that failed.

---

## Running the golden set evaluation

The golden set covers the full range of question difficulty. Running it against your deployed system gives you a regression baseline:

```bash
# Run full evaluation
python evaluation/run_eval.py

# Output shows per-question pass/fail and aggregate score
# Example output:
# GS-001 PASS  relevance=0.87  coverage=0.92  unsupported=0.05
# GS-016 PASS  (out-of-scope correctly declined)
# GS-025 FAIL  coverage=0.48  (cross-document check failed — review chunk size)
```

When you change the chunker, embedder, or retrieval parameters, re-run the golden set to catch regressions before they reach users.
