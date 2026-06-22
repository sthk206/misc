# TableRAG: A Retrieval Augmented Generation Framework for Heterogeneous Document Reasoning

Repo for _[TableRAG: A Retrieval Augmented Generation Framework for Heterogeneous Document Reasoning](https://github.com/yxh-y/TableRAG/)_  

![Main Architecture](./figures/Main%20structure.png)

# 📌 Introduction

- We identify two key limitations of existing RAG approaches in the context of heterogeneous document question answering: structural information loss and lack of global view. 
- We propose **TableRAG**, an **Hybrid (SQL Execution and Textual Retrieval) framework** that unifies textual understanding and complex manipulations over tabular data. TableRAG comprises an offline database construction phase and a four-step online iterative reasoning process.
- We develop **HeteQA**, a benchmark for evaluating multi-hop heterogeneous reasoning capabilities. Experimental results show that TableRAG outperforms RAG and programmatic approaches on HeteQA and public benchmarks, establishing a state-of-the-art solution.

# 🔎 Setup

## Environment
```
conda create -n your_env python=3.10

git clone https://github.com/yxh-y/TableRAG/
cd TableRAG

pip install -r requirements.txt
```

# 🛠 How to Run?

## Dataset Preparation
1. Download dev_excel.zip from [Google Drive](https://drive.google.com/drive/folders/1Pea6kiUZv0UP8k7Ohv19KorBdBaUrouE?usp=drive_link).

## Offline Workflow

### Step 1: Setup MySQL Database

1. Download MySQL
Reach https://downloads.mysql.com/archives/community/ and find MySQL 8.0.24 and downloads for your appropriate environment.

2. Install MySQL
```
tar -zxvf mysql-8.0.24-linux-glibc2.12-x86_64.tar.gz
cd mysql-8.0.24-linux-glibc2.12-x86_64
sudo mkdir /usr/local/mysql && sudo mv * /usr/local/mysql/
sudo groupadd mysql
sudo useradd -r -g mysql mysql
cd /usr/local/mysql
sudo bin/mysqld --initialize --user=mysql --basedir=/usr/local/mysql --datadir=/usr/local/mysql/data
sudo cp support-files/mysql.server /etc/init.d/mysql
sudo systemctl enable mysql
sudo systemctl start mysql
```
3. Create Database for TableRAG
```sql
CREATE DATABASE TableRAG;
```

### Step 2: Offline data Ingestion

1. Setup database config 
Edit offline_data_ingestion_and_query_interface/config/database_config.json and update it with your own MySQL config.

2. Prepare table files to be ingested
Unzip dev_excel.zip to 'offline_data_ingestion_and_query_interface/dataset/hybridqa/dev_excel/'.

4. Execute data ingestion pipeline
```
cd offline_data_ingestion_and_query_interface/src/
python data_persistent.py
```

### Step 3: Start Database query service

1. Setup LLM config
Edit 'offline_data_ingestion_and_query_interface/src/handle_requests.py' and substitute your llm request url and apikey into model_request_config.

2. Start service to provide SQL query interface

```
python interface.py
```

## Online Workflow

### Step 1: Setup Config and Data Source

1. Edit 'online_inference/config.py' to set the LLM infering url and key, and the query service url.
   
3. Unzip the dev_excel.zip and put it into "/data" directory.

### Step 2: Run Main Experiment
```
cd online_inference
python3 main.py
  --backbone <backbone_llm>
  --data_file_path ./data/my_dev.json
  --save_file_path <path to save file>
  --max_iter <max iterations of TableRAG, default to 5>
  --rerun <True if some cases fail at the previous run, default to False> 
```





# poc

 Locked decisions

 - System 2 fidelity: faithful local reimplementation of TableRAG's method (iterative
 decomposition + NL→SQL over structured tables + text retrieval). SQLite instead of
 MySQL;
 no GPU/local-model dependency.
 - Parser: pdfplumber primary, Camelot (lattice) fallback. Realistic end-to-end
 ("Option A"): TableRAG is fed the auto-parsed tables, noise included — parser errors
 legitimately count against it (they map to the "table parsing failure" category).
 - Ground truth: human-verified from the source PDF, never taken from the auto-parse
 (avoids grading the parser against itself).
 - LLM + embeddings: behind a placeholder OpenAI-compatible gateway the user wires up
 later. User supplies get_bearer_token(); we build OpenAI(api_key=token, base_url=...)
 and
 call client.chat.completions.create(...) and client.embeddings.create(model, input).
 Model names are placeholder constants.
 - Env: create a venv, install what's needed, update requirements.txt.
 - Shared across both systems (fairness): same gateway LLM, same embedding model, same
 chunking (1000/200), same FAISS top-k retrieval. No neural reranker either side
 (simplification,
 documented). The only differences are System 2's structured-table store + NL→SQL +
 iterative decomposition vs. baseline's text-only single-shot retrieve→generate.

 Layout (all new code under poc_eval/, repo untouched elsewhere)

 poc_eval/
   common/llm_gateway.py        # placeholder client + chat()/embed() wrappers + model
 constants
   ingestion/extract_pdf.py     # per-page text (for D) + table extraction (for T) over
 selected pages
   baseline_rag/pipeline.py     # chunk → embed → FAISS → top-k → generate (tables seen
 as linearized text)
   tablerag/pipeline.py         # D text retrieval + T structured store (SQLite) + NL→SQL
 + iterative decomposition
   tablerag/sql_store.py        # load parsed tables into SQLite, schema JSON
   benchmark/benchmark_questions.json
   run_eval.py                  # run both systems on every Q, score, classify failures
   results/evaluation_results.csv
   results/summary_report.md
   README.md

 Implementation steps

 1. Env setup. python3 -m venv .venv; install pdfplumber, camelot-py[cv] (+ brew install
 ghostscript
 for Camelot), faiss-cpu, openai, langchain-text-splitters, pandas, numpy. Pin into
 requirements.txt.
 2. Gateway shim common/llm_gateway.py: get_bearer_token() placeholder (raises
 NotImplementedError
 with a comment showing where the user drops their function); get_client() →
 OpenAI(api_key=get_bearer_token(), base_url=GATEWAY_URL);
 chat(messages, tools=None, temperature=0.1) and embed(texts). Constants CHAT_MODEL,
 EMBED_MODEL, GATEWAY_URL (placeholders).
 3. Section selection. Read the 10-K TOC (now that pdfplumber is installed), pick the
 table-dense
 sections (Derivatives, Market Risk/VaR, Sensitivity, Trading Assets/Liabilities, Credit
 Risk,
 Fair Value). Target ~10–30 pages total. Record the page ranges.
 4. Ingestion ingestion/extract_pdf.py over selected pages only: (a) page text → for
 baseline;
 (b) tables via pdfplumber (extract_tables), Camelot lattice fallback → JSON records with
 page, inferred title (nearest caption above bbox), headers/schema, rows. Persist tables
 JSON.
 5. Benchmark benchmark/benchmark_questions.json: author ~15 questions across
 A (exact retrieval), B (comparison), C (arithmetic/aggregation), D
 (cross-table/multi-step).
 Each entry: id, category, question, answer, pages[], table_title, supporting_evidence,
 reasoning.
 Verify every answer by manual inspection of the source PDF text (not the auto-parse).
 6. Baseline RAG baseline_rag/pipeline.py: RecursiveCharacterTextSplitter(1000/200) over
 page
 text (tables appear only as whatever linearized text the PDF yields — no table
     6. Baseline RAG baseline_rag/pipeline.py: RecursiveCharacterTextSplitter(1000/200)
     over page
     text (tables appear only as whatever linearized text the PDF yields — no table
     handling),
     embed via gateway, FAISS IndexFlatIP, retrieve top-5, single-shot generate. Return
     {answer, retrieved_chunks, cited_pages}.
     7. TableRAG tablerag/: load parsed tables into SQLite (sql_store.py); online loop
     mirrors
     online_inference/main.py — LLM decomposes query into subqueries (tool-calls), table
     subqueries
     → retrieve relevant table by embedding title/schema → NL→SQL → execute on SQLite →
     results;
     text subqueries → same text retrieval as baseline; synthesize; iterate max_iter=5.
     Return
     {answer, retrieved_tables, sql, sql_results, retrieved_chunks, cited_pages,
     iterations}.
     8. Eval harness run_eval.py: run both systems per question; score 4 axes — answer
     correctness
     (LLM-judge, reusing the repo's online_inference/evaluation/hybrid_eval.py llm_eval
     pattern via
     the gateway), numeric correctness (programmatic exact match w/ unit normalization),
     evidence
     correctness (did it surface the right table), source correctness (right page). For
     each wrong
     answer, classify into the taxonomy (retrieval / table-parsing / row-col / arithmetic
     / reasoning /
     hallucination) via heuristics + LLM-assist. Write results/evaluation_results.csv.
     9. Report results/summary_report.md: benchmark overview; per-category accuracy for
     both systems;
     qualitative analysis (where baseline won/lost, where TableRAG added value or didn't);
     conservative
     conclusion — no superiority claim unless the numbers support it. Document
     parser-as-confound and
     the SQLite/no-reranker simplifications as caveats.

     Reused / referenced repo assets

     - online_inference/main.py — the 4-step iterative agent loop to mirror.
     - online_inference/tools/sql_tool.py — NL→SQL execution pattern (adapt MySQL→SQLite).
     - online_inference/tools/retriever.py — chunking constants (1000/200), recall/rerank
     shape.
     - online_inference/evaluation/hybrid_eval.py + prompt.py — llm_eval judging pattern.

     Verification (end-to-end)

     1. poc_eval/README.md documents wiring get_bearer_token() + setting GATEWAY_URL/model
     constants.
     2. Smoke test: run extract_pdf.py on the selected pages; eyeball 1–2 parsed tables
     vs. PDF.
     3. Run run_eval.py --limit 3 (both systems on 3 questions) once the gateway is wired
     — confirm both
     return answers + evidence + the CSV row schema is correct.
     4. Full run → inspect evaluation_results.csv and summary_report.md.
     5. Sanity: confirm baseline and TableRAG used identical model/embedding/chunk
     settings (printed at startup).

     Out of scope (documented as caveats)

     Whole-document coverage; hedge-vs-speculation judgment; MySQL/Flask service; local
     bge-m3/reranker
     - GPU; neural reranking; statistical significance (n≈15).

     Done. The full POC is built, compiles, and runs end-to-end (validated in offline mock
  mode). Here's what landed.

  What was built — poc_eval/

  A self-contained evaluation comparing Baseline RAG vs TableRAG on corp-10k-2025.pdf
  (JPMorgan 2025 10-K, 410 pp).

  Pipeline (all runnable now):
  - ingestion/extract_pdf.py + table_parser.py — extracts page text + tables from the 29
  selected table-dense pages. Stock pdfplumber/Camelot collapsed on these borderless
  tables (the credit table parsed as 2×4), so I built a custom whitespace/numeric parser —
  automated and uniform, true to Option A. Produced 36 tables; every key value verified
  correct (Swaps→19,056; Total notional→50,642; Total VaR→[40,25,136,…]; credit
  Subtotal→1,446,098).
  - baseline_rag/pipeline.py — chunk(1000/200)→embed→FAISS→single-shot generate.
  - tablerag/{sql_store,pipeline}.py — tables→SQLite + NL→SQL + iterative sub-query loop,
  mirroring online_inference/main.py. SQL aggregation works (IR % of total = 58.32%,
  matching ground truth).
  - common/ — shared gateway shim, FAISS retriever, JSON util, mock gateway. Both systems
  share LLM/embeddings/chunking/retrieval; the only difference is the structured-table +
  SQL path.
  - run_eval.py + report.py — runs both systems on all 16 Qs, scores 4 axes
  (answer/numeric/evidence/source), classifies failures into your 6-category taxonomy,
  writes evaluation_results.csv + summary_report.md (conservative conclusion logic).

  Deliverables present: benchmark/benchmark_questions.json (16 Qs, A/B/C/D, ground truth
  hand-verified from the PDF), results/evaluation_results.csv, results/summary_report.md.

  What you need to do to get real numbers

  Two edits in poc_eval/common/llm_gateway.py: implement get_bearer_token() (your token
  fn) and set GATEWAY_URL / CHAT_MODEL / EMBED_MODEL (or the POC_* env vars). Then:

  source .venv/bin/activate
  python -m poc_eval.run_eval          # full run → results/

  Honest caveats baked in

  - The committed results/* are from a --mock run (banner-marked, all-zero) — they only
  demonstrate output format. Real numbers require the gateway.
  - Option A: parser noise counts against TableRAG (it's a confound, documented).
  - n=16, no significance testing; SQLite/gateway-embeddings/no-reranker diverge from the
  paper's exact stack to stay runnable and fair.

  The earlier finding stands and is captured in the report: the repo ships an Excel→MySQL
  ingestion, not the paper's PDF front-end, so we built that front-end ourselves.

  One thing I should flag honestly: I cannot report whether TableRAG actually wins — that
  requires running against your gateway. The conclusion in summary_report.md will populate
  automatically from the real CSV when you run it.