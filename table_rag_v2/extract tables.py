"""
Two-tier PDF table extractor for dense financial documents (e.g. 10-K filings).

Logic
-----
1. Camelot stream  -> first pass. Free, fast, wins dense regular grids.
                      Each detected table is run through a subtotal-reconciliation
                      gate (do the numeric rows sum to a labelled Subtotal/Total?).
2. img2table       -> second pass, ONLY on pages where Camelot found nothing or
                      produced tables that fail the gate. Robust on side-by-side
                      and narrative-beside-table layouts that fragment Camelot.

Returns
-------
list[tuple[str, pandas.DataFrame]]
    Each tuple is ("p{page} :: {source} :: {gate}", dataframe).
    e.g. ("p130 :: camelot :: reconciled", <df>)
"""

from __future__ import annotations
import re
import warnings
from typing import List, Tuple, Optional

import pandas as pd

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------- helpers

_NUM_RE = re.compile(r"^\(?\$?\s*-?[\d,]+(?:\.\d+)?\)?$")
_TOTAL_WORDS = ("total", "subtotal")


def _to_number(cell: object) -> Optional[float]:
    """Parse a financial cell to float. Returns None for non-numeric / blank / dash."""
    if cell is None:
        return None
    s = str(cell).strip()
    if s in ("", "—", "-", "–", "nan", "NaN"):
        return None
    s = s.replace("$", "").replace(",", "").replace("\n", " ").strip()
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()").strip()
    if not s or not re.match(r"^-?\d+(\.\d+)?$", s):
        return None
    val = float(s)
    return -val if neg else val


def _first_numeric_col(df: pd.DataFrame) -> Optional[int]:
    """Index of the column with the most parseable numbers (the primary value column)."""
    best, best_count = None, 0
    for c in range(df.shape[1]):
        cnt = sum(_to_number(v) is not None for v in df.iloc[:, c])
        if cnt > best_count:
            best, best_count = c, cnt
    return best if best_count >= 3 else None


def _reconciles(df: pd.DataFrame, tol: float = 0.02) -> Optional[bool]:
    """
    Subtotal-reconciliation gate.

    Looks for a row labelled Subtotal/Total in the label column and checks whether
    the non-total numeric rows above it sum to it (within `tol` relative tolerance)
    in the primary numeric column.

    Returns True  -> a total was found and the column sums to it (high confidence)
            False -> a total was found but the sum does NOT match (likely bad parse)
            None  -> no total row found, gate is not applicable (can't judge)
    """
    if df is None or df.empty or df.shape[1] < 2:
        return None
    col = _first_numeric_col(df)
    if col is None:
        return None
    label_col = 0 if col != 0 else (1 if df.shape[1] > 1 else 0)

    parts, total_val = [], None
    for _, row in df.iterrows():
        label = str(row.iloc[label_col]).strip().lower()
        val = _to_number(row.iloc[col])
        if val is None:
            continue
        if any(w in label for w in _TOTAL_WORDS):
            total_val = val          # last total wins (grand total)
        else:
            parts.append(val)
    if total_val is None or not parts:
        return None

    # A grand "Total" often = Subtotal + reconciling items, so test the best subset:
    # accept if the running sum of the leading block hits the total.
    target = total_val
    for cut in range(len(parts), 0, -1):
        if abs(sum(parts[:cut]) - target) <= max(abs(target) * tol, 1.0):
            return True
    # also accept exact full-sum match
    return abs(sum(parts) - target) <= max(abs(target) * tol, 1.0)


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop fully-empty rows/cols and strip whitespace."""
    df = df.copy()
    df = df.map(lambda x: str(x).strip() if x is not None else "")
    df = df.loc[:, (df != "").any(axis=0)]
    df = df.loc[(df != "").any(axis=1)]
    return df.reset_index(drop=True)


def _looks_like_table(df: pd.DataFrame) -> bool:
    """
    Sanity filter. Accept a frame as a table if it has >=2 rows and >=2 cols
    and EITHER has a numeric backbone (>=3 parseable numbers) OR looks like a
    consistent text grid (most rows fill most columns -> aligned tabular data,
    e.g. a Name/Title/Date roster where only one column is numeric).
    """
    if df is None or df.shape[0] < 2 or df.shape[1] < 2:
        return False
    numeric = sum(_to_number(v) is not None for v in df.values.ravel())
    if numeric >= 3:
        return True
    # text-grid heuristic: rows that fill at least half their columns
    filled_rows = sum(
        (sum(1 for c in row if str(c).strip()) >= max(2, df.shape[1] // 2))
        for _, row in df.iterrows()
    )
    return filled_rows >= 3 and df.shape[1] >= 3


def _prose_contaminated(df: pd.DataFrame, max_share: float = 0.15) -> bool:
    """
    Detect narrative text leaking into a table (the Camelot side-by-side /
    narrative-beside-table failure mode). A cell counts as 'prose' if it has
    many words and ends like a sentence. If too many rows are prose-heavy,
    the parse is structurally unreliable even if it reconciles numerically.
    """
    if df is None or df.empty:
        return False
    prose_rows = 0
    for _, row in df.iterrows():
        joined = " ".join(str(c) for c in row).strip()
        words = joined.split()
        # long, sentence-like, and lacking the numeric backbone of a data row
        has_numbers = any(_to_number(c) is not None for c in row)
        if len(words) >= 12 and not has_numbers:
            prose_rows += 1
        elif len(words) >= 20:        # very long even with a stray number
            prose_rows += 1
    return prose_rows / len(df) > max_share


def _side_by_side(df: pd.DataFrame) -> bool:
    """
    Detect two tables merged horizontally (Camelot's side-by-side failure).
    Fingerprint: a non-numeric label repeats within a single row IN THE HEADER
    REGION (first few rows) -- e.g. 'Employees ... Employees' or
    'Total Firm ... Total Firm'. Restricting to the top rows avoids false
    positives from labels that happen to repeat deep in a long single table.
    """
    if df is None or df.shape[1] < 4:
        return False
    header_region = df.head(min(5, len(df)))
    for _, row in header_region.iterrows():
        labels = [str(c).strip().lower() for c in row
                  if str(c).strip() and _to_number(c) is None]
        # keep only real word-labels: drop pure punctuation/symbols like '$', '%',
        # '(7) %' etc. A repeated '$' across the 2025/2024 columns is NOT a sign of
        # two side-by-side tables.
        labels = [s for s in labels if re.search(r"[a-z]{3,}", s)]
        if len(labels) >= 2 and len(labels) != len(set(labels)):
            return True
    return False


def _split_side_by_side(df: pd.DataFrame) -> List[pd.DataFrame]:
    """
    If a frame is two tables merged horizontally (duplicated header across the
    left and right halves), split it at the midpoint into separate frames.
    Returns [df] unchanged if no clean split is found.
    """
    if df is None or df.shape[1] < 4 or not _side_by_side(df):
        return [df]
    ncol = df.shape[1]
    # find a split point where the left block's header equals the right block's
    header = [str(c).strip().lower() for c in df.iloc[min(1, len(df) - 1)]]
    # only treat alphabetic header labels as meaningful for the seam test
    def _wordy(s: str) -> str:
        return s if re.search(r"[a-z]{3,}", s) else ""
    header = [_wordy(h) for h in header]
    for cut in range(2, ncol - 1):
        left, right = header[:cut], header[cut:]
        # align by comparing the leading min(len) labels of each half
        m = min(len(left), len(right))
        if m >= 2 and left[:m] == right[:m]:
            a = df.iloc[:, :cut].reset_index(drop=True)
            b = df.iloc[:, cut:].reset_index(drop=True)
            return [a, b]
    # fallback: split at the exact midpoint for even column counts
    if ncol % 2 == 0:
        half = ncol // 2
        return [df.iloc[:, :half].reset_index(drop=True),
                df.iloc[:, half:].reset_index(drop=True)]
    return [df]


# ----------------------------------------------------------------------------- tier 1

def _camelot_page(pdf_path: str, page: int) -> List[pd.DataFrame]:
    import camelot
    out = []
    try:
        tables = camelot.read_pdf(pdf_path, pages=str(page), flavor="stream")
        for t in tables:
            df = _clean(t.df)
            if _looks_like_table(df):
                out.append(df)
    except Exception:
        pass
    return out


# ----------------------------------------------------------------------------- tier 2

def _img2table_page(pdf_path: str, page: int) -> List[pd.DataFrame]:
    from img2table.document import PDF
    from img2table.ocr import TesseractOCR
    out = []
    try:
        doc = PDF(pdf_path, pages=[page - 1])          # img2table is 0-indexed
        res = doc.extract_tables(
            ocr=TesseractOCR(lang="eng"),
            implicit_rows=True, implicit_columns=True,
            borderless_tables=True, min_confidence=40,
        )
        for tbls in res.values():
            for t in tbls:
                df = _clean(t.df)
                if not _looks_like_table(df):
                    continue
                for part in _split_side_by_side(df):   # split merged side-by-sides
                    if _looks_like_table(part):
                        out.append(part)
    except Exception:
        pass
    return out


# ----------------------------------------------------------------------------- driver

def extract_tables(
    pdf_path: str,
    pages: Optional[List[int]] = None,
) -> List[Tuple[str, pd.DataFrame]]:
    """
    Extract tables from a PDF using the two-tier strategy.

    Parameters
    ----------
    pdf_path : str
    pages    : optional list of 1-based PDF page numbers. If None, scans all pages
               that pdfplumber flags as containing a table (keeps it cheap).

    Returns
    -------
    list of (label, dataframe). label encodes page, source engine, and gate result.
    """
    import pdfplumber

    if pages is None:
        pages = []
        with pdfplumber.open(pdf_path) as pdf:
            for i, pg in enumerate(pdf.pages, start=1):
                # default find_tables misses borderless tables, so also probe
                # with the text strategy and a simple aligned-rows signal.
                found = bool(pg.find_tables())
                if not found:
                    try:
                        tt = pg.find_tables({
                            "vertical_strategy": "text",
                            "horizontal_strategy": "text",
                        })
                        # require a few multi-cell rows to avoid flagging prose
                        found = any(
                            t.bbox and (t.bbox[2] - t.bbox[0]) > 200
                            and len(t.rows) >= 3 and len(t.columns) >= 3
                            for t in tt
                        )
                    except Exception:
                        found = False
                if found:
                    pages.append(i)

    results: List[Tuple[str, pd.DataFrame]] = []

    for page in pages:
        # --- Tier 1: Camelot stream
        cam = _camelot_page(pdf_path, page)
        cam_gated = [(df, _reconciles(df)) for df in cam]

        # Escalate to img2table only for STRUCTURAL failures that img2table can
        # actually fix: narrative prose leaking in, or two tables merged
        # side-by-side. A clean parse that merely can't be reconciled (e.g. a
        # balance sheet with intermediate subtotals, where summing rows double-
        # counts) should be KEPT as Camelot output, not thrown away.
        structural_fail = bool(cam) and any(
            _prose_contaminated(df) or _side_by_side(df) for df in cam
        )

        # keep Camelot if it found something and isn't structurally broken;
        # a pure gate failure on a clean parse is allowed through as 'unverified'
        page_ok = bool(cam) and not structural_fail

        if page_ok:
            for df, g in cam_gated:
                tag = {True: "reconciled", False: "failed-gate", None: "unverified"}[g]
                results.append((f"p{page} :: camelot :: {tag}", df))
            continue

        # --- Tier 2: img2table fallback
        img = _img2table_page(pdf_path, page)
        if img:
            for df in img:
                g = _reconciles(df)
                tag = {True: "reconciled", False: "failed-gate", None: "unverified"}[g]
                results.append((f"p{page} :: img2table :: {tag}", df))
        elif cam:
            # img2table found nothing; fall back to whatever Camelot had, flagged
            for df, g in cam_gated:
                tag = {True: "reconciled", False: "failed-gate", None: "unverified"}[g]
                results.append((f"p{page} :: camelot(fallback) :: {tag}", df))

    return results


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "corp-10k-2024.pdf"
    test_pages = [10, 53, 92, 130, 248]
    out = extract_tables(path, pages=test_pages)
    print(f"\nExtracted {len(out)} table(s) from pages {test_pages}\n")
    for label, df in out:
        print(f"### {label}  shape={df.shape}")
        print(df.head(6).to_string(max_cols=10))
        print()
