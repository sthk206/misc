"""
Structured table store for TableRAG.

Loads the auto-parsed tables (data/tables.json) into an in-memory SQLite database --
one SQL table per parsed table -- so the agent can run real SQL (filters, MAX, SUM,
ratios) over the figures. SQLite stands in for the repo's MySQL backend; the NL->SQL
+ execute pattern mirrors online_inference/tools/sql_tool.py.

Numeric cells are stored as REAL (NULL for em-dash / NM / unparsed), so aggregation
and comparison work directly. The original row label is kept in `row_label`.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from typing import Any

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TABLES_PATH = os.path.join(ROOT, "poc_eval", "data", "tables.json")


def _sql_name(table_id: str) -> str:
    return "t_" + re.sub(r"[^0-9a-zA-Z_]", "_", table_id)


class SQLStore:
    def __init__(self, tables: list[dict[str, Any]] | None = None):
        if tables is None:
            with open(TABLES_PATH) as f:
                tables = json.load(f)
        self.tables = tables
        self.conn = sqlite3.connect(":memory:")
        self.registry: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        cur = self.conn.cursor()
        for t in self.tables:
            name = _sql_name(t["table_id"])
            ncol = t["n_value_cols"]
            cols = ["row_label TEXT"] + [f"c{i+1} REAL" for i in range(ncol)]
            cur.execute(f'CREATE TABLE "{name}" ({", ".join(cols)})')
            placeholders = ", ".join(["?"] * (ncol + 1))
            for r in t["rows"]:
                vals = [r["label"]] + [
                    (v if isinstance(v, (int, float)) else None) for v in r["values"]
                ]
                vals += [None] * (ncol + 1 - len(vals))
                cur.execute(f'INSERT INTO "{name}" VALUES ({placeholders})', vals[: ncol + 1])
            self.registry[name] = {
                "sql_name": name,
                "table_id": t["table_id"],
                "title": t["title"],
                "header_context": t["header_context"],
                "section": t["section"],
                "pdf_page": t["pdf_page"],
                "columns": ["row_label"] + [f"c{i+1}" for i in range(ncol)],
                "row_labels": [r["label"] for r in t["rows"]],
                "sample_rows": t["rows"][:6],
            }
        self.conn.commit()

    def schema_doc(self, sql_name: str) -> str:
        """Human/LLM-readable description so NL->SQL knows what the c-columns mean."""
        info = self.registry[sql_name]
        lines = [
            f'TABLE "{sql_name}"  (pdf_page {info["pdf_page"]}, section {info["section"]})',
            f'  title: {info["title"]!r}',
            f'  columns: {", ".join(info["columns"])}',
            "  column meaning (from the table header, left->right after row_label):",
        ]
        for h in info["header_context"]:
            lines.append(f"    {h}")
        lines.append("  sample rows:")
        for r in info["sample_rows"]:
            cells = " | ".join(
                f"c{i+1}={'' if v is None else v}" for i, v in enumerate(r["values"])
            )
            lines.append(f"    row_label={r['label']!r} | {cells}")
        return "\n".join(lines)

    def search_doc(self, sql_name: str) -> str:
        """Compact text used to embed/retrieve this table."""
        info = self.registry[sql_name]
        return (
            f"{info['title']}\n{' '.join(info['header_context'])}\n"
            f"rows: {', '.join(info['row_labels'])}"
        )

    def execute(self, sql: str) -> tuple[list[str], list[list[Any]]]:
        cur = self.conn.cursor()
        cur.execute(sql)
        cols = [d[0] for d in cur.description] if cur.description else []
        rows = [list(r) for r in cur.fetchall()]
        return cols, rows
