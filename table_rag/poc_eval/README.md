# TableRAG vs. Baseline RAG — Proof-of-Concept Evaluation

A small, honest POC answering one question:

> **Does TableRAG provide a measurable advantage over baseline RAG when answering
> questions that depend on structured financial tables?**

Document under test: `../corp-10k-2025.pdf` (JPMorgan Chase 2025 Form 10-K, 410 pages).
This is **not** a publishable benchmark — it is a focused, ~16-question stress test of the
table-dependent capabilities (exact retrieval, comparison, arithmetic, cross-table
reasoning) that underlie harder downstream tasks like hedge-vs-speculation analysis.

## Two design facts worth knowing first

1. **Paper vs. released code.** The TableRAG *paper* describes ingestion starting from a
   PDF (split into text `D` and tables `T`, with table parsing → schema → DB). The released
   repo does **not** ship that PDF front-end — its offline ingestion reads *pre-extracted*
   Excel (`pd.read_excel`) into MySQL. So to run TableRAG's *method* on a real 10-K we build
   the PDF→table front-end ourselves (`ingestion/`), and reimplement the online method
   (`tablerag/`) faithfully but locally (SQLite, gateway embeddings, portable JSON action
   protocol instead of MySQL/bge/tool-calls).

2. **Option A (realistic end-to-end).** TableRAG is fed the **auto-parsed** tables, parser
   noise included; parser errors legitimately count against it. Benchmark ground truth, by
   contrast, was **hand-verified from the source PDF**, never from the parser — so we never
   grade the parser against itself.

## Fairness

Both systems share, via `common/` + the gateway: the same LLM, the same embedding model,
the same chunking (1000/200), and the same FAISS top-k retrieval. The **only** difference is:

| | Baseline RAG (System 1) | TableRAG (System 2) |
|---|---|---|
| Tables | seen only as linearized page text | parsed into a SQLite store |
| Reasoning | single-shot retrieve → generate | iterative sub-query decomposition + NL→SQL + text retrieval |

## Layout

```
poc_eval/
  common/llm_gateway.py     # <-- WIRE YOUR GATEWAY HERE
  common/retrieval.py       # shared FAISS dense retriever
  common/jsonutil.py        # LLM-JSON parsing
  common/mock_gateway.py    # offline fake gateway (plumbing checks only)
  config/sections.json      # the table-dense pages selected from the 10-K
  ingestion/extract_pdf.py  # PDF -> pages.json (text) + tables.json (parsed tables)
  ingestion/table_parser.py # custom financial-table parser (borderless tables)
  baseline_rag/pipeline.py  # System 1
  tablerag/sql_store.py     # tables.json -> in-memory SQLite + schema docs
  tablerag/pipeline.py      # System 2 (agent loop)
  benchmark/benchmark_questions.json   # 16 questions + verified ground truth
  run_eval.py               # runs both systems, scores, classifies failures
  report.py                 # builds summary_report.md from results
  results/                  # evaluation_results.csv + summary_report.md
  requirements.txt
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate     # from repo root
pip install -r poc_eval/requirements.txt
```
(Verified to install/import on Python 3.14, macOS, no GPU.)

## Wire up the gateway (2 things)

Edit `common/llm_gateway.py`:

1. Implement `get_bearer_token()` — drop in your existing token function. Its return value
   is passed straight to `OpenAI(api_key=<token>, base_url=GATEWAY_URL)`.
2. Set `GATEWAY_URL`, `CHAT_MODEL`, `EMBED_MODEL` (or export `POC_GATEWAY_URL`,
   `POC_CHAT_MODEL`, `POC_EMBED_MODEL`; a quick token can be exported as `POC_GATEWAY_TOKEN`).

Generation uses `client.chat.completions.create(...)`; embeddings use
`client.embeddings.create(model, input)`. Nothing else needs changing.

## Run

```bash
# from repo root, with the venv active
python -m poc_eval.ingestion.extract_pdf      # (re)build pages.json + tables.json
python -m poc_eval.run_eval                    # full run -> results/ (needs the gateway)
python -m poc_eval.run_eval --limit 3          # quick smoke test
python -m poc_eval.run_eval --mock             # offline plumbing check, no key/network
```

## Deliverables

- `benchmark/benchmark_questions.json` — 16 questions (A/B/C/D) with answer, page(s), table
  title, supporting evidence, and reasoning.
- `results/evaluation_results.csv` — one row per (question × system): answers, values, cited
  & evidence pages, the four correctness flags, failure type, iterations, SQL used.
- `results/summary_report.md` — overview, per-category accuracy for both systems, qualitative
  analysis, conservative conclusion, caveats. Auto-regenerated from the CSV on every run.

> The `results/*` files currently committed are from a `--mock` run (clearly banner-marked)
> and exist only to show the output format. **Re-run against the real gateway to populate
> meaningful numbers.**

## Scoring & failure taxonomy

Four axes per question: **answer correctness** (LLM judge, with an exact-numeric shortcut),
**numeric correctness** (key number within 1%), **evidence correctness** (correct supporting
table/page retrieved), **source correctness** (correct page cited). Every wrong answer is
classified into one of: retrieval / table-parsing / row-column / arithmetic / reasoning /
hallucination.

## Caveats

Small sample (n=16, no significance testing); Option A means parser quality is a confound for
TableRAG; SQLite/gateway-embeddings/no-reranker diverge from the paper's exact stack to stay
runnable and fair; out of scope = whole-document coverage and subjective hedge-vs-speculation
judgment.
