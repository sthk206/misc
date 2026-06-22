"""
Custom financial-table parser.

Stock pdfplumber/Camelot table detection relies on ruling lines and collapses on
the borderless, right-aligned numeric tables typical of SEC filings (verified: the
JPM 10-K credit-exposure table parses to 2x4 with defaults). We therefore build our
own parser, as agreed for this POC.

It is still fully AUTOMATED and applied UNIFORMLY -- no human types in values and no
per-table hand-tuning -- so it preserves the "Option A / realistic end-to-end"
contract: whatever it gets wrong counts against TableRAG as a table-parsing failure.

Approach: financial tables here are `<row label>  <number> <number> ...` with
right-aligned numeric columns. We tokenize each text line into a label plus a list
of numeric cells, group consecutive numeric lines into table blocks, and attach the
preceding non-numeric lines as the title / column-header context.
"""

from __future__ import annotations

import re
from typing import Any, Optional

# A numeric cell: parenthesized negative, plain number (with thousands separators /
# decimals), an em dash (null placeholder), or NM ("not meaningful").
_NUM = re.compile(r"\(\d[\d,]*(?:\.\d+)?\)|-?\d[\d,]*(?:\.\d+)?|—|NM")
_YEAR = re.compile(r"^\d{4}$")


def _looks_like_period_header(cells: list[str]) -> bool:
    """True if >=2 cells are 4-digit years (e.g. '2025 2024' or 'December 31 2025 2024').
    These are column-period headers, not data rows."""
    years = 0
    for c in cells:
        c = c.replace(",", "")
        if _YEAR.match(c) and 2018 <= int(c) <= 2031:
            years += 1
    return years >= 2


def to_number(tok: str) -> Optional[float]:
    tok = tok.strip()
    if tok in ("—", "NM", ""):
        return None
    neg = tok.startswith("(") and tok.endswith(")")
    body = tok.strip("()").replace(",", "")
    try:
        v = float(body)
    except ValueError:
        return None
    return -v if neg else v


def tokenize(line: str) -> tuple[str, list[str]]:
    """Split a line into (row_label, [numeric cell tokens])."""
    clean = line.replace("$", " ")
    matches = list(_NUM.finditer(clean))
    if not matches:
        return line.strip(), []
    label = clean[: matches[0].start()].strip()
    cells = [m.group(0) for m in matches]
    return label, cells


def parse_tables(page_text: str, min_rows: int = 2) -> list[dict[str, Any]]:
    """Parse all numeric table blocks from a page's text."""
    raw_lines = [ln for ln in page_text.split("\n")]
    parsed = []
    for ln in raw_lines:
        label, cells = tokenize(ln)
        # A data row has >=2 numeric cells and is not a column-period header
        # (e.g. "2025 2024"). NOTE: do not reject merely because cells are 4-digit;
        # financial values like 3,305 are 4 digits too -- only true years count.
        is_data = len(cells) >= 2 and not _looks_like_period_header(cells)
        parsed.append({"text": ln.strip(), "label": label, "cells": cells, "is_data": is_data})

    tables: list[dict[str, Any]] = []
    i = 0
    n = len(parsed)
    while i < n:
        if not parsed[i]["is_data"]:
            i += 1
            continue
        # Grow a block of (mostly) data lines; tolerate single non-data separators.
        start = i
        j = i
        gap = 0
        last_data = i
        while j < n:
            if parsed[j]["is_data"]:
                last_data = j
                gap = 0
            else:
                gap += 1
                if gap > 2:  # tolerate sub-headers / blank lines within a table
                    break
            j += 1
        block = parsed[start : last_data + 1]
        data_rows = [b for b in block if b["is_data"]]
        # Discard prose blocks (footnotes / narrative with embedded numbers): real
        # table rows have short labels; sentence fragments do not.
        if data_rows:
            frac_long = sum(len(b["label"].split()) > 9 for b in data_rows) / len(data_rows)
        else:
            frac_long = 1.0
        if len(data_rows) >= min_rows and frac_long <= 0.5:
            # Header context = up to 4 non-data lines immediately preceding the block.
            header_ctx = []
            k = start - 1
            while k >= 0 and len(header_ctx) < 4 and not parsed[k]["is_data"]:
                header_ctx.insert(0, parsed[k]["text"])
                k -= 1
            n_cols = max(len(b["cells"]) for b in data_rows)
            rows = []
            for b in data_rows:
                vals = [to_number(c) for c in b["cells"]]
                vals += [None] * (n_cols - len(vals))
                rows.append({"label": b["label"], "values": vals, "raw_cells": b["cells"]})
            title = _pick_title(header_ctx)
            tables.append(
                {
                    "title": title,
                    "header_context": header_ctx,
                    "n_value_cols": n_cols,
                    "rows": rows,
                }
            )
        i = last_data + 1
    return tables


def _pick_title(header_ctx: list[str]) -> str:
    """Heuristic: the title is usually the last short, non-sentence header line."""
    if not header_ctx:
        return ""
    for line in reversed(header_ctx):
        s = line.strip()
        if not s:
            continue
        # Column-period lines / unit lines are not titles.
        if re.search(r"\b20\d\d\b", s) or s.lower().startswith("(in ") or "(in millions" in s.lower():
            continue
        if len(s) <= 80:
            return s
    return header_ctx[-1].strip()


def to_markdown(table: dict[str, Any]) -> str:
    ncol = table["n_value_cols"]
    head = ["row_label"] + [f"c{i+1}" for i in range(ncol)]
    out = ["| " + " | ".join(head) + " |", "| " + " | ".join(["---"] * len(head)) + " |"]
    for r in table["rows"]:
        cells = [r["label"]] + [("" if v is None else _fmt(v)) for v in r["values"]]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def _fmt(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else str(v)
