"""
Version 1 -- "auto" parser.

Uses pdfplumber (a pure-pip, no-Java table parser; the most appropriate library here
given the env) to extract EVERY table it can find across the whole 10-K, with no
hand-tuning. This is the realistic baseline: financial tables in a 10-K are dense,
multi-header, and footnoted, so the raw extraction is frequently messy (merged
headers, split rows, blank columns). That messiness is exactly the point of the
comparison against the "perfect" version.

Each detected table -> a cleaned DataFrame -> one .xlsx. Tables too small to be useful
(<2 rows or <2 cols of content) are skipped.
"""
from __future__ import annotations

from typing import List, Tuple

import pandas as pd
import pdfplumber
from tqdm import tqdm

from poc_eval.parse.table_utils import clean_columns, transfer_name


def _table_to_df(raw: List[List]) -> pd.DataFrame | None:
    # Drop fully-empty rows/cols.
    rows = [[(c or "").strip() for c in row] for row in raw]
    rows = [r for r in rows if any(cell for cell in r)]
    if len(rows) < 2:
        return None
    header, *body = rows
    # Give empty header cells positional names so they survive.
    header = [h if h else f"col{i}" for i, h in enumerate(header)]
    width = len(header)
    body = [r + [""] * (width - len(r)) if len(r) < width else r[:width] for r in body]
    df = pd.DataFrame(body, columns=header)
    # Need at least 2 columns of real content to be a usable table.
    if df.shape[1] < 2 or df.shape[0] < 1:
        return None
    return df


def extract_auto_tables(pdf_path: str) -> List[Tuple[str, pd.DataFrame]]:
    out: List[Tuple[str, pd.DataFrame]] = []
    seen_names: dict = {}
    with pdfplumber.open(pdf_path) as pdf:
        for idx, page in enumerate(tqdm(pdf.pages, desc="auto-tables")):
            pdf_page = idx + 1
            try:
                tables = page.extract_tables()
            except Exception:
                continue
            for tidx, raw in enumerate(tables):
                df = _table_to_df(raw)
                if df is None:
                    continue
                base = transfer_name(f"page{pdf_page}_table{tidx}")
                # de-dup defensively
                if base in seen_names:
                    seen_names[base] += 1
                    base = f"{base}_{seen_names[base]}"
                else:
                    seen_names[base] = 0
                out.append((base, clean_columns(df)))
    return out
