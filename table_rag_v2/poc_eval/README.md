# poc_eval — TableRAG vs. Baseline RAG on a real 10-K

A self-contained, **CPU-only** harness that swaps the repo's `dev_excel.zip` data source
for **`corp-10k-2025.pdf`** (JPMorgan Chase 2025 Form 10-K) and compares three systems on
the 16-question benchmark in `benchmark/benchmark_questions.json`.

All LLM + embedding calls go through **`common/llm_gateway.py`** (one shim, identical
config for every system, so the comparison is fair). Because embeddings/chat run on the
gateway, **no GPU and no local bge models are needed**; `faiss-cpu` replaces the GPU index.

## The three systems

| System | Tables | Retrieval | SQL | Notes |
|---|---|---|---|---|
| `baseline-rag` | — | text chunks only | no | plain RAG control; reads page text, cites pages |
| `tablerag-auto` | pdfplumber auto-extract (228 tables, messy) | tables-as-markdown + text | yes | realistic parser baseline |
| `tablerag-perfect` | 4 hand-perfect tables | tables-as-markdown + text | yes | upper bound: assumes a flawless parser |

`tablerag-*` reuse the repo's pipeline **verbatim** — same prompts (sourced directly from
`online_inference/prompt.py` and the offline `prompt.py` via `systems/repo_prompts.py`),
same agent loop (decompose → per-subquery retrieve + NL2SQL + COMBINE → answer), same
schema/NL2SQL/MySQL execution. Only the LLM/embedding backend (gateway) and data source
(this PDF) differ.

## The two parser versions (D + T)

The PDF is split into the repo's two modalities:
- **D (document text):** every page's text → `data/<version>/doc/page_XXXX.json`. Identical
  across versions. (`parse/pdf_text.py`)
- **T (tables) → `.xlsx`:**
  - **auto** (`parse/auto_tables.py`): `pdfplumber` extracts *every* table in the 10-K, no
    tuning — dense financial tables come out messy. This is the realistic case.
  - **perfect** (`parse/perfect_tables.py`): the 4 benchmarked tables (pdf pages 122, 138,
    207, 214) hand-transcribed into clean, typed DataFrames. Values transcribed from the
    PDF; the benchmark ground truth was independently hand-verified against the PDF, so this
    is not grading the parser against itself.

Both feed the repo-compatible flow: `.xlsx` → `excel_to_markdown` (retrieval) **and** →
MySQL + schema JSON (SQL). This is the answer to "can we call `load_hybrid_dataset` with the
PDF + extracted tables?" — almost: the PDF first becomes per-page doc JSON (D) and per-table
`.xlsx` (T), which is exactly what the repo's loader consumes.

## Setup

```bash
# 1. deps (CPU-only)
python -m venv .venv && source .venv/bin/activate
pip install -r poc_eval/requirements.txt

# 2. MySQL (the only system dependency)
brew install mysql && mysql.server start      # fresh install: user=root, no password

# 3. (real runs only) wire the gateway: edit poc_eval/common/llm_gateway.py
#    - implement get_bearer_token()
#    - set GATEWAY_URL / CHAT_MODEL / EMBED_MODEL (or POC_* env vars)
```

DB connection / paths are configurable via `POC_*` env vars (see `config.py`); defaults
match a fresh brew MySQL. Each version gets its own database (`tablerag_poc_auto`,
`tablerag_poc_perfect`).

## Run

```bash
# end-to-end (build datasets → ingest → eval)
bash poc_eval/build_all.sh            # real gateway
bash poc_eval/build_all.sh --mock     # offline plumbing check, no creds/network

# or step by step
python -m poc_eval.parse.build_dataset --version perfect
python -m poc_eval.parse.build_dataset --version auto
python -m poc_eval.ingest.mysql_ingest --version perfect
python -m poc_eval.ingest.mysql_ingest --version auto
python -m poc_eval.run_eval --mock                 # add --limit N / --systems ...
```

### Mock vs. real
`--mock` installs `common/mock_gateway.py`, which returns **canned** replies and
bag-of-words embeddings. It exercises the full pipeline (parse → ingest → retrieve → agent
→ SQL → score → report) with no API key, but **its metrics are plumbing artifacts, not
findings** (the judge always returns 0, etc.). Run against the real gateway for real numbers.

## Outputs (`poc_eval/results/<timestamp>/`)
- `traces.jsonl` — one line per (system, question): the **full trace** — subqueries issued,
  files retrieved at each step, generated SQL + execution result, subquery answers.
- `summary.json` — per-system accuracy, per-category (A/B/C/D) breakdown, failure counts.
- `report.md` — accuracy table + per-question ✅/❌ grid + failure breakdown.

Inspect where a question failed across systems:
```bash
python -m poc_eval.show_trace --question C1
```

## Layout
```
poc_eval/
  config.py              paths, MySQL config, retrieval/agent knobs
  parse/                 PDF → D (doc JSON) + T (xlsx) ; auto & perfect ; schema JSON
  ingest/mysql_ingest.py xlsx → per-version MySQL database
  systems/
    retriever.py         gateway-embedding + faiss-cpu retriever (repo MixedDocRetriever)
    nl2sql.py            NL2SQL → MySQL (repo offline service)
    tablerag.py          agent loop with full tracing (repo main._run)
    baseline_rag.py      plain text RAG control
    repo_prompts.py      repo prompts, loaded verbatim
  eval/judge.py          LLM-judge score + failure taxonomy
  run_eval.py            orchestrate 3 systems × benchmark → traces + report
  show_trace.py          pretty-print a question's trace across systems
  common/                llm_gateway.py (real) + mock_gateway.py (offline)
```
