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

## System 2 = the repo's actual TableRAG (not a reimplementation)

`tablerag/runner.py` instantiates and runs the repo's real `online_inference/main.py`
`TableRAG` — its 4-step loop, `SYSTEM_EXPLORE_PROMPT` / `COMBINE_PROMPT`, `solve_subquery`
tool-calls, `MixedDocRetriever` recall→rerank flow, and the offline NL2SQL prompt + schema
generation. We only change what an OpenAI-compatible **gateway** (no local models/GPU) and
"no servers" force; the prompts and control flow are the repo's, verbatim.

Forced adaptations (each marked `# POC:` in the patched repo files):
- **bge-m3 embedder → gateway embeddings** (`utils/tool_utils.Embedder`); **bge-reranker
  dropped → pass-through** that preserves recall order (no gateway rerank endpoint).
- **GPU faiss → CPU** `IndexFlatIP` (`tools/retriever.py`).
- **DeepSeek/gpt-4o config → gateway** `llm_config` (`config.py`); the repo's `get_chat_result`
  is routed through `llm_gateway.chat`, so the bearer token flows into `OpenAI(...)` exactly
  as the repo does.
- **MySQL + Flask NL2SQL service → in-process SQLite** (`tools/sql_tool.py`), reusing the
  offline NL2SQL prompts + `extract_sql_statement` + the pandas schema-gen verbatim.
- **Tables kept as JSON** (no xlsx): `build_corpus.py` turns each table into a DataFrame and
  feeds the repo's exact schema-gen / SQLite insert / retriever-markdown logic.
- A couple of shipped bugs in `main.py._run` (a wrong variable reference + a bad tuple unpack
  in the SQL block) are fixed minimally so the loop runs; the algorithm is unchanged.

**Two deliberate, paper-aligned choices about retrieval (not the repo's defaults):**
- **Open retrieval** — we drop the repo's `query + "The given table is in {table_id}"` suffix.
  The repo names the gold table in the query (HeteQA queries are built that way); our 10-K
  questions don't name a table, so the system must *find* it. This makes retrieval harder, not
  easier — a conservative choice.
- **Per-sub-query, top-k table selection (paper eq. 4)** instead of the repo's frozen top-1.
  The repo's top-1 worked *because* it had the table-name hint; without that hint, freezing on
  top-1 would unfairly starve the SQL step. So each sub-query runs NL2SQL over the schemas of
  **all table chunks in its top-k retrieved set** (`S_t`), matching the paper's *"for each chunk
  in the top-ranked set … extract its associated schema."* (`main.py._run`, marked `# POC`.)

Because System 2 is the repo verbatim, it returns only an answer string; the runner captures
the pages it actually retrieved (for evidence scoring) and parses the key number from the
answer (for numeric scoring). It does not emit explicit page citations, so "source
correctness" for TableRAG is taken as its retrieved-evidence pages.

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
  ingestion/build_gold_tables.py  # -> data/gold_tables.json (verified tables, Option B)
  baseline_rag/pipeline.py  # System 1 (no equivalent in the repo, so written here)
  tablerag/build_corpus.py  # our JSON tables -> repo corpus (per-table JSON + schema JSON + SQLite)
  tablerag/runner.py        # System 2 = drives the REPO's online_inference/main.py TableRAG
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
python -m poc_eval.ingestion.build_gold_tables # build the verified gold tables (Option B)
python -m poc_eval.run_eval                    # Option A: full run -> results/ (needs gateway)
python -m poc_eval.run_eval --clean            # Option B: TableRAG fed verified gold tables
python -m poc_eval.run_eval --limit 3          # quick smoke test
python -m poc_eval.run_eval --mock             # offline plumbing check, no key/network
```

### Option A vs. Option B (the parser sensitivity check)

- **Option A** (`run_eval`, default): TableRAG consumes the **auto-parsed** tables — realistic
  end-to-end; parser errors count against it. → `results/evaluation_results.csv`, `summary_report.md`.
- **Option B** (`run_eval --clean`): TableRAG is fed **`data/gold_tables.json`** — hand-verified
  clean versions of the 4 tables the benchmark depends on (built by `build_gold_tables.py` from
  the source PDF text, independent of the parser). → `results/evaluation_results_optionB.csv`,
  `summary_report_optionB.md`.

Run both, then read the TableRAG accuracy in each: **the A→B gap is the damage attributable to
PDF table parsing**, separated from the value of the method itself. If A ≈ B, parsing isn't the
bottleneck; if B ≫ A, TableRAG's method works but our parser is holding it back. Everything else
(retrieval, NL→SQL, agent loop, the baseline) is identical across the two runs.

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
