"""
System 2: TableRAG (faithful local reimplementation of the method).

Mirrors the repo's online_inference/main.py iterative agent: the LLM decomposes the
question into sub-queries; for each sub-query we (a) retrieve the most relevant
structured tables and run NL->SQL over them (online_inference/tools/sql_tool.py
pattern, executed on the local SQLite store), and (b) retrieve text chunks (the D
branch). Observations are fed back and the loop repeats up to max_iter before the
model emits a final answer.

Differences from the repo, all to run locally without GPU/keys (documented):
  - SQLite instead of MySQL.
  - Gateway embeddings instead of local bge-m3 (shared with the baseline for fairness).
  - A portable JSON action protocol instead of the OpenAI tool-calls API, so it works
    against any OpenAI-compatible gateway. The control flow (decompose -> retrieve ->
    SQL -> synthesize, iterated) is the same.
  - Tables are the AUTO-PARSED ones (Option A): parser noise can and will surface here.
"""

from __future__ import annotations

import json
import os
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from poc_eval.baseline_rag.pipeline import CHUNK_OVERLAP, CHUNK_SIZE, PAGES_PATH
from poc_eval.common import llm_gateway
from poc_eval.common.jsonutil import extract_json
from poc_eval.common.retrieval import VectorIndex
from poc_eval.tablerag.sql_store import SQLStore

MAX_ITER = 5            # matches the repo's default/hard limit
TABLE_TOPK = 3
TEXT_TOPK = 3

CONTROLLER_SYSTEM = (
    "You are TableRAG, a financial-analysis agent answering a question about a 10-K "
    "filing. You work iteratively. On each turn respond with ONE JSON object and nothing "
    "else.\n"
    "To gather evidence:  {\"action\":\"query\",\"subquery\":\"<one focused question to "
    "look up in the filing's tables/text>\"}\n"
    "When you have enough evidence:  {\"action\":\"final\",\"answer\":\"<concise answer "
    "with units>\",\"value\":<the single key number, plain, no commas/units, or null>,"
    "\"pages\":[<PDF page numbers used>]}\n"
    "Decompose multi-step or arithmetic questions into sub-queries. Prefer letting the "
    "SQL results do the arithmetic. Do not invent numbers; rely on the observations."
)

NL2SQL_SYSTEM = (
    "You translate a question into ONE SQLite SELECT over the given tables. Tables hold "
    "financial figures; numeric columns are named c1, c2, ... and `row_label` holds the "
    "row name. Use the column-meaning notes to map periods/metrics to the right c-column. "
    "Use LIKE for row matching. You may use SUM, MAX, MIN, and arithmetic. "
    "Respond with ONE JSON object: {\"table\":\"<table name>\",\"sql\":\"<single SELECT>\"}."
)


class TableRAG:
    name = "tablerag"

    def __init__(self) -> None:
        self.store = SQLStore()
        # Table retrieval index (T branch): embed title + headers + row labels.
        table_docs = [
            {"sql_name": n, "pdf_page": info["pdf_page"], "text": self.store.search_doc(n)}
            for n, info in self.store.registry.items()
        ]
        self.table_index = VectorIndex(table_docs, text_key="text")

        # Text retrieval index (D branch): identical chunking to the baseline.
        with open(PAGES_PATH) as f:
            pages = json.load(f)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        chunks = []
        for pg in pages:
            for ci, piece in enumerate(splitter.split_text(pg["text"])):
                chunks.append(
                    {
                        "chunk_id": f"p{pg['pdf_page']}_c{ci}",
                        "pdf_page": pg["pdf_page"],
                        "text": f"[PDF page {pg['pdf_page']}]\n{piece}",
                    }
                )
        self.text_index = VectorIndex(chunks, text_key="text")

    # ---- sub-query handling -----------------------------------------------------
    def _run_sql(self, subquery: str, tables: list[dict[str, Any]]) -> dict[str, Any]:
        schema = "\n\n".join(self.store.schema_doc(t["sql_name"]) for t in tables)
        messages = [
            {"role": "system", "content": NL2SQL_SYSTEM},
            {"role": "user", "content": f"Tables:\n{schema}\n\nQuestion: {subquery}"},
        ]
        raw = llm_gateway.chat_text(messages)
        spec = extract_json(raw)
        sql = (spec.get("sql") or "").strip()
        out: dict[str, Any] = {"sql": sql, "table": spec.get("table"), "rows": [], "error": None}
        if not sql:
            out["error"] = "no SQL produced"
            return out
        try:
            cols, rows = self.store.execute(sql)
            out["columns"] = cols
            out["rows"] = rows[:25]
        except Exception as e:  # noqa: BLE001 - surface any SQLite error to the agent
            out["error"] = str(e)
        return out

    def _observe(self, subquery: str) -> dict[str, Any]:
        tables = self.table_index.search(subquery, k=TABLE_TOPK)
        chunks = self.text_index.search(subquery, k=TEXT_TOPK)
        sql_res = self._run_sql(subquery, tables) if tables else {"error": "no tables"}
        return {"subquery": subquery, "tables": tables, "chunks": chunks, "sql": sql_res}

    @staticmethod
    def _format_observation(obs: dict[str, Any]) -> str:
        lines = [f"Observation for sub-query: {obs['subquery']!r}"]
        sql = obs.get("sql", {})
        if sql.get("sql"):
            lines.append(f"  SQL ({sql.get('table')}): {sql['sql']}")
        if sql.get("error"):
            lines.append(f"  SQL error: {sql['error']}")
        elif sql.get("rows") is not None:
            lines.append(f"  SQL result columns: {sql.get('columns')}")
            for r in sql.get("rows", [])[:15]:
                lines.append(f"    {r}")
        if obs.get("chunks"):
            lines.append("  Text snippets:")
            for c in obs["chunks"]:
                snippet = c["text"].replace("\n", " ")[:300]
                lines.append(f"    [p{c['pdf_page']}] {snippet}")
        return "\n".join(lines)

    # ---- main loop --------------------------------------------------------------
    def answer(self, question: str) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": CONTROLLER_SYSTEM},
            {"role": "user", "content": f"Question: {question}"},
        ]
        observations: list[dict[str, Any]] = []
        sql_log: list[dict[str, Any]] = []
        tables_used: list[dict[str, Any]] = []
        iters = 0
        final: dict[str, Any] = {}

        for iters in range(1, MAX_ITER + 1):
            raw = llm_gateway.chat_text(messages)
            act = extract_json(raw)
            if act.get("action") == "final" or "answer" in act:
                final = act
                break
            subquery = act.get("subquery") or question
            obs = self._observe(subquery)
            observations.append(obs)
            for t in obs.get("tables", []):
                tables_used.append({"sql_name": t["sql_name"], "pdf_page": t["pdf_page"]})
            if obs.get("sql"):
                sql_log.append(
                    {"subquery": subquery, "sql": obs["sql"].get("sql"),
                     "error": obs["sql"].get("error"), "rows": obs["sql"].get("rows")}
                )
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": self._format_observation(obs)})
        else:
            # ran out of iterations: force a final answer from accumulated evidence
            messages.append(
                {"role": "user",
                 "content": "Provide your final answer now as the JSON final action."}
            )
            final = extract_json(llm_gateway.chat_text(messages))

        evidence_pages = sorted({t["pdf_page"] for t in tables_used} |
                                {c["pdf_page"] for o in observations for c in o.get("chunks", [])})
        return {
            "system": self.name,
            "question": question,
            "answer": final.get("answer", "NOT FOUND"),
            "value": final.get("value"),
            "cited_pages": final.get("pages", []),
            "iterations": iters,
            "retrieved_evidence": [
                {"sql_name": t["sql_name"], "pdf_page": t["pdf_page"]}
                for t in _dedupe(tables_used)
            ],
            "evidence_pages": evidence_pages,
            "sql_log": sql_log,
            "raw_response": json.dumps(final),
        }


def _dedupe(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    out = []
    for it in items:
        key = it["sql_name"]
        if key not in seen:
            seen.add(key)
            out.append(it)
    return out
