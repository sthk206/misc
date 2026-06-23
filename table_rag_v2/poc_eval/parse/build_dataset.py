"""
Build a complete on-disk dataset for one parser version.

Produces, under poc_eval/data/<version>/:
  excel/<table>.xlsx   -- the T (table) modality, one workbook per table
  schema/<table>.json  -- {table_name, column_list} consumed by NL2SQL
  doc/page_XXXX.json   -- the D (document-text) modality, one per PDF page

The doc/ text is identical across versions; only excel/ + schema/ differ
(messy auto-extraction vs. clean perfect transcription).

Usage:
  python -m poc_eval.parse.build_dataset --version auto
  python -m poc_eval.parse.build_dataset --version perfect
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

from poc_eval import config
from poc_eval.parse import auto_tables, perfect_tables, pdf_text
from poc_eval.parse.table_utils import build_schema, infer_and_convert, transfer_name


def _write_tables(tables, version: str) -> int:
    xdir, sdir = config.excel_dir(version), config.schema_dir(version)
    os.makedirs(xdir, exist_ok=True)
    os.makedirs(sdir, exist_ok=True)
    written = 0
    for raw_name, df in tables:
        table_name = transfer_name(raw_name)
        try:
            df_conv = df.apply(infer_and_convert)
        except Exception:
            df_conv = df
        try:
            df_conv.to_excel(os.path.join(xdir, f"{table_name}.xlsx"), index=False)
        except Exception as e:  # a badly-shaped auto table; skip it
            print(f"  skip {table_name}: {e}")
            continue
        schema = build_schema(df_conv, table_name)
        with open(os.path.join(sdir, f"{table_name}.json"), "w", encoding="utf-8") as f:
            json.dump(schema, f, ensure_ascii=False)
        written += 1
    return written


def build(version: str, with_text: bool = True) -> dict:
    assert version in ("auto", "perfect"), version
    # Start clean so reruns don't accumulate stale tables.
    if os.path.isdir(config.excel_dir(version)):
        shutil.rmtree(config.excel_dir(version))
    if os.path.isdir(config.schema_dir(version)):
        shutil.rmtree(config.schema_dir(version))

    if version == "auto":
        tables = auto_tables.extract_auto_tables(config.SOURCE_PDF)
    else:
        tables = perfect_tables.extract_perfect_tables()
    n_tables = _write_tables(tables, version)

    n_docs = 0
    if with_text:
        n_docs = pdf_text.extract_doc_text(config.SOURCE_PDF, config.doc_dir(version))

    summary = {"version": version, "tables": n_tables, "doc_pages": n_docs}
    print(f"[build_dataset] {summary}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=["auto", "perfect"], required=True)
    ap.add_argument("--no-text", action="store_true", help="skip doc-text extraction")
    args = ap.parse_args()
    build(args.version, with_text=not args.no_text)
