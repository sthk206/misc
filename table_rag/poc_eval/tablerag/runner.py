"""
Runner: drives the REPO's TableRAG (online_inference/main.py) on our PDF-derived corpus.

It does NOT reimplement the agent -- it instantiates the repo's `TableRAG` and calls its
`_run`, so the loop, SYSTEM_EXPLORE_PROMPT/COMBINE_PROMPT, solve_subquery tools, retriever
flow, and NL2SQL are the repo's. The only wiring here is the forced gateway/SQLite/JSON setup:
  - route the repo's get_chat_result through the gateway (poc_eval.common.llm_gateway.chat),
  - point config_mapping["gateway"] at the gateway url/model/token,
  - build the corpus (build_corpus) and configure the in-process SQLite NL2SQL tool,
  - capture retrieved filenames -> pages for the evaluator (the repo emits only an answer).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "online_inference"))

from poc_eval.common import llm_gateway          # noqa: E402
from poc_eval.tablerag.build_corpus import build  # noqa: E402

AUTO_TABLES = os.path.join(ROOT, "poc_eval", "data", "tables.json")
PAGES = os.path.join(ROOT, "poc_eval", "data", "pages.json")


def _safe_token() -> str:
    try:
        return llm_gateway.get_bearer_token()
    except Exception:
        return "unused-in-mock"  # mock patches the chat path, so the token is never used


def _patch_repo_llm():
    """Route the repo's get_chat_result through the gateway shim (forced LLM swap)."""
    import chat_utils
    import main as repo_main

    def get_chat_result(messages, tools=None, tool_choice=None, llm_config=None):
        return llm_gateway.chat(messages, tools=tools, tool_choice=tool_choice)

    chat_utils.get_chat_result = get_chat_result
    repo_main.get_chat_result = get_chat_result  # main imported it via `from chat_utils import *`
    return repo_main


def _parse_number(text: str):
    m = re.search(r'-?\$?\s*\(?\d[\d,]*(?:\.\d+)?\)?', text or "")
    if not m:
        return None
    s = m.group(0).replace("$", "").replace(",", "").replace(" ", "")
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()")
    try:
        v = float(s)
        return -v if neg else v
    except ValueError:
        return None


class TableRAGRunner:
    """Adapter exposing the repo's TableRAG as `.answer(question) -> dict` for run_eval."""

    name = "tablerag"

    def __init__(self, tables_path: str | None = None) -> None:
        import config as repo_config
        import tools.sql_tool as sql_tool

        repo_config.config_mapping["gateway"] = {
            "url": llm_gateway.GATEWAY_URL, "model": llm_gateway.CHAT_MODEL,
            "api_key": _safe_token(),
        }

        tables = json.load(open(tables_path or AUTO_TABLES))
        pages = json.load(open(PAGES))
        out_dir = os.path.join(ROOT, "poc_eval", "data",
                               "corpus_gold" if tables_path else "corpus_auto")
        paths = build(tables, pages, out_dir)
        sql_tool.configure(paths["sqlite_path"], paths["schema_dir"])
        self.page_of = {t["table_id"]: t["pdf_page"] for t in tables}

        repo_main = _patch_repo_llm()
        args = argparse.Namespace(doc_dir=paths["doc_dir"], excel_dir=paths["excel_dir"],
                                  bge_dir="", max_iter=5)
        self.tr = repo_main.TableRAG(args)

        # Capture retrieved filenames (the repo's TableRAG returns only an answer string).
        self._filenames: list[str] = []
        _orig = self.tr.retriever.retrieve

        def _wrapped(query, recall_num, rerank_num):
            docs, scores, filenames = _orig(query, recall_num, rerank_num)
            self._filenames.extend(filenames)
            return docs, scores, filenames

        self.tr.retriever.retrieve = _wrapped

    def _page_from_filename(self, fn: str):
        name = fn.replace(".json", "")
        m = re.match(r"page_(\d+)$", name)
        if m:
            return int(m.group(1))
        return self.page_of.get(name)

    def answer(self, question: str) -> dict:
        self._filenames = []
        try:
            ans, _messages = self.tr._run({"question": question}, backbone="gateway")
        except Exception as e:  # noqa: BLE001
            ans = f"ERROR: {e}"
        ans = (ans or "NOT FOUND").strip()
        pages = sorted({p for fn in self._filenames if (p := self._page_from_filename(fn))})
        return {
            "system": self.name,
            "question": question,
            "answer": ans,
            "value": _parse_number(ans),
            # repo TableRAG emits no explicit citations; use the pages it actually retrieved.
            "cited_pages": pages,
            "evidence_pages": pages,
            "retrieved_evidence": [{"filename": fn} for fn in dict.fromkeys(self._filenames)],
            "sql_log": [],
            "iterations": None,
            "raw_response": ans,
        }
