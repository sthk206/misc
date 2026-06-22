"""
PDF ingestion for the POC.

Produces two artifacts over the *selected* table-dense pages only (see
config/sections.json):

  data/pages.json   -- per-page text. Feeds the BASELINE RAG corpus (D). Tables
                       appear here only as whatever linearized text the PDF yields;
                       no table-aware processing. This is the point of the baseline.

  data/tables.json  -- auto-parsed tables (pdfplumber primary; Camelot lattice as an
                       optional fallback). Feeds the TableRAG structured store (T).
                       Per "Option A" (realistic end-to-end) we keep whatever the
                       parser produces, noise and all -- parser errors legitimately
                       count against TableRAG in the failure taxonomy.

Run:  python -m poc_eval.ingestion.extract_pdf
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Any

import pdfplumber

from poc_eval.ingestion.table_parser import parse_tables, to_markdown

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SECTIONS_PATH = os.path.join(ROOT, "poc_eval", "config", "sections.json")
DATA_DIR = os.path.join(ROOT, "poc_eval", "data")

PRINTED_OFFSET = 2  # pdf physical page = printed footer page + 2


def _load_sections() -> dict[str, Any]:
    with open(SECTIONS_PATH) as f:
        return json.load(f)


def _page_to_section(sections: dict[str, Any]) -> dict[int, str]:
    """Map each selected pdf page -> section name."""
    out: dict[int, str] = {}
    for sec in sections["selected_sections"]:
        for p in sec["pdf_pages"]:
            out[p] = sec["name"]
    return out


def extract() -> tuple[list[dict], list[dict]]:
    sections = _load_sections()
    page_section = _page_to_section(sections)
    selected = sorted(page_section.keys())

    pdf_path = os.path.join(ROOT, sections["source_pdf"])
    pages_out: list[dict] = []
    tables_out: list[dict] = []

    with pdfplumber.open(pdf_path) as pdf:
        for pno in selected:
            page = pdf.pages[pno - 1]
            section = page_section[pno]
            printed = pno - PRINTED_OFFSET

            text = page.extract_text() or ""
            pages_out.append(
                {"pdf_page": pno, "printed_page": printed, "section": section, "text": text}
            )

            for ti, tbl in enumerate(parse_tables(text)):
                tables_out.append(
                    {
                        "table_id": f"p{pno}_t{ti}",
                        "pdf_page": pno,
                        "printed_page": printed,
                        "section": section,
                        "title": tbl["title"],
                        "header_context": tbl["header_context"],
                        "rows": tbl["rows"],
                        "n_rows": len(tbl["rows"]),
                        "n_value_cols": tbl["n_value_cols"],
                        "markdown": to_markdown(tbl),
                        "parser": "custom_financial",
                    }
                )

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, "pages.json"), "w") as f:
        json.dump(pages_out, f, indent=2, ensure_ascii=False)
    with open(os.path.join(DATA_DIR, "tables.json"), "w") as f:
        json.dump(tables_out, f, indent=2, ensure_ascii=False)
    return pages_out, tables_out


def main() -> None:
    pages, tables = extract()
    print(f"Extracted {len(pages)} pages and {len(tables)} tables.")
    by_sec: dict[str, int] = defaultdict(int)
    for t in tables:
        by_sec[t["section"]] += 1
    for sec, n in by_sec.items():
        print(f"  {sec}: {n} tables")
    print("\nSample parsed table titles:")
    for t in tables[:12]:
        print(f"  [{t['table_id']} p{t['pdf_page']} {t['n_rows']}x{t['n_value_cols']}] {t['title'][:70]!r}")


if __name__ == "__main__":
    main()
