# poc_eval — TableRAG vs. Baseline RAG on a real 10-K

A self-contained, **CPU-only** harness that swaps the repo's `dev_excel.zip` data source
for **`corp-10k-2025.pdf`** (JPMorgan Chase 2025 Form 10-K) and compares five systems on
the 16-question benchmark in `benchmark/benchmark_questions.json`.

All LLM + embedding calls go through **`common/llm_gateway.py`** (one shim, identical
config for every system, so the comparison is fair). Because embeddings/chat run on the
gateway, **no GPU and no local bge models are needed**; `faiss-cpu` replaces the GPU index.

## The five systems

Two axes: parser quality (auto vs. perfect) × table hint (off vs. on), plus the baseline.

| System | Tables | SQL | Table hint | Notes |
|---|---|---|---|---|
| `baseline-rag` | — | no | n/a | plain RAG control; reads page text, cites pages |
| `tablerag-auto` | pdfplumber auto-extract (228 tables, messy) | yes | off | realistic parser, must find the table |
| `tablerag-auto-hint` | pdfplumber auto-extract (228 tables, messy) | yes | **on** | realistic parser, gold table hinted |
| `tablerag-perfect` | 4 hand-perfect tables | yes | off | flawless parser, must find the table |
| `tablerag-perfect-hint` | 4 hand-perfect tables | yes | **on** | flawless parser, gold table hinted |

`tablerag-*` reuse the repo's pipeline **verbatim** — same prompts (sourced directly from
`online_inference/prompt.py` and the offline `prompt.py` via `systems/repo_prompts.py`),
same agent loop (decompose → per-subquery retrieve + NL2SQL + COMBINE → answer), same
schema/NL2SQL/MySQL execution. Only the LLM/embedding backend (gateway) and data source
(this PDF) differ. (See "Fidelity" below for the precise deviations.)

### Table hint
The repo seeds table selection with the gold table id:
`query + f"The given table is in {table_id}"` (`online_inference/main.py:133`). The
benchmark here has no `table_id`, but each question carries a `table_title`, so the `-hint`
variants append `f"The given table is in {table_title}"` before the top-1 table retrieval
(and only there, matching the repo — per-subquery retrieval is never hinted). The hint-off
variants must find the right table from the bare question, which is the harder, more
realistic setting. Running both isolates how much oracle table selection is worth.

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

Add `--mock` for an offline plumbing check (no creds/network); drop it for real numbers.
Add `--limit N` to run only the first N questions.

```bash
# end-to-end (build datasets → ingest → eval all 5 systems)
bash poc_eval/build_all.sh            # real gateway
bash poc_eval/build_all.sh --mock     # offline plumbing check

# or step by step
python -m poc_eval.parse.build_dataset --version perfect
python -m poc_eval.parse.build_dataset --version auto
python -m poc_eval.ingest.mysql_ingest --version perfect
python -m poc_eval.ingest.mysql_ingest --version auto
python -m poc_eval.run_eval --mock                 # all 5 systems (default)
```

### Run systems individually
`--systems` accepts one or more of: `baseline-rag`, `tablerag-auto`, `tablerag-auto-hint`,
`tablerag-perfect`, `tablerag-perfect-hint`.

```bash
python -m poc_eval.run_eval --mock --systems baseline-rag
python -m poc_eval.run_eval --mock --systems tablerag-auto
python -m poc_eval.run_eval --mock --systems tablerag-auto-hint
python -m poc_eval.run_eval --mock --systems tablerag-perfect
python -m poc_eval.run_eval --mock --systems tablerag-perfect-hint
```

### Useful A/B groupings
```bash
# how much is oracle table selection worth? (hint off vs on, perfect tables)
python -m poc_eval.run_eval --mock --systems tablerag-perfect tablerag-perfect-hint

# how much does a perfect parser buy? (auto vs perfect, hint off)
python -m poc_eval.run_eval --mock --systems tablerag-auto tablerag-perfect
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
  run_eval.py            orchestrate 5 systems × benchmark → traces + report
  show_trace.py          pretty-print a question's trace across systems
  common/                llm_gateway.py (real) + mock_gateway.py (offline)
```

## Fidelity (how close to the original repo?)
The `systems/` modules **re-implement** the repo's logic rather than importing its
functions — the repo modules aren't import-safe (GPU/torch at import time, hardcoded
endpoints) and `main._run` has runtime bugs. The prompts are loaded from the repo
verbatim; everything else mirrors the repo with these deliberate deviations:

- **Embeddings/chat** go through the gateway, not local `bge-m3` on GPU.
- **No cross-encoder reranker** — the repo reranks recall with `bge-reranker-v2-m3`; here
  `retrieve` returns the top-k by embedding score (the gateway exposes no reranker).
- **Table selection** is by retrieval (optionally `-hint`ed); the repo is given `table_id`.
- **Judge prompt** uses an "impartial grader" rubric (the `mock_gateway` contract) instead
  of the repo's `EVALUATION_PRONPT`; same 0/1 semantics.
- **dtype mapping** simplified (BIGINT/DOUBLE/TEXT) vs. the repo's `dtype_mapping.py`.
- **Seed table render** uses `excel_to_markdown` (we have `.xlsx`) vs. the repo's
  `read_plain_csv`.

`baseline-rag` has no repo counterpart (it's the control for this comparison).
