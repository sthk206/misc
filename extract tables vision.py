"""
Vision-based PDF table extractor.

Same signature and return type as the rule-based extractor:

    extract_tables_vision(pdf_path, client, model, pages=None)
        -> list[tuple[str, pandas.DataFrame]]

Each page is rasterized to a PNG and sent to a vision model through an
OpenAI-style client (chat.completions.create with image_url content blocks).
The model returns JSON describing every table on the page; we parse it into
DataFrames. No table detection, cropping, or column-naming is required from
the caller -- the model reads headers and separates multiple tables itself.

The `client` is any object exposing:
    client.chat.completions.create(model=..., messages=..., **kw)
        -> response with response.choices[0].message.content (a string)
i.e. the standard OpenAI Python SDK surface. A gateway client that mimics
this works unchanged.
"""

from __future__ import annotations
import base64
import io
import json
import re
from typing import List, Tuple, Optional

import pandas as pd


# --------------------------------------------------------------------------- prompt

_SYSTEM = (
    "You are a precise table-extraction engine for financial documents. "
    "You receive an image of a single PDF page and return only JSON."
)

_INSTRUCTION = """\
Extract EVERY table visible in this page image.

Return ONLY valid JSON (no prose, no markdown fences) with this shape:

{
  "tables": [
    {
      "title": "<the table's title or caption, or null>",
      "columns": ["<header 1>", "<header 2>", ...],
      "rows": [
        ["<cell>", "<cell>", ...],
        ...
      ]
    }
  ]
}

Rules:
- If the page has no table, return {"tables": []}.
- Treat two tables placed side by side as TWO separate table objects.
- Flatten multi-level headers with " / " (e.g. "Selected metrics / Net charge-offs").
- Preserve values exactly as printed: keep parentheses for negatives, keep
  thousands separators, keep "%" and "$". Use null for an empty cell or an
  em-dash (—).
- Every row must have the same number of entries as "columns".
- Do not include footnote paragraphs or narrative text as table rows.
"""


# --------------------------------------------------------------------------- raster

def _render_page_png(pdf_path: str, page_1based: int, dpi: int = 150) -> bytes:
    """Render one PDF page to PNG bytes using pypdfium2 (no image files needed)."""
    import pypdfium2 as pdfium
    pdf = pdfium.PdfDocument(pdf_path)
    try:
        page = pdf[page_1based - 1]
        bitmap = page.render(scale=dpi / 72)
        pil = bitmap.to_pil()
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        return buf.getvalue()
    finally:
        pdf.close()


def _png_to_data_url(png: bytes) -> str:
    b64 = base64.b64encode(png).decode()
    return f"data:image/png;base64,{b64}"


# --------------------------------------------------------------------------- model call

def _call_vision(client, model: str, png: bytes) -> str:
    """Send the page image to the vision model, return the raw text response."""
    messages = [
        {"role": "system", "content": _SYSTEM},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": _INSTRUCTION},
                {"type": "image_url",
                 "image_url": {"url": _png_to_data_url(png)}},
            ],
        },
    ]
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        max_tokens=4096,
    )
    return resp.choices[0].message.content


# --------------------------------------------------------------------------- parsing

def _extract_json(text: str) -> Optional[dict]:
    """Parse the model's JSON, tolerating stray prose or ```json fences."""
    if not text:
        return None
    s = text.strip()
    # strip code fences if present
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    try:
        return json.loads(s)
    except Exception:
        pass
    # fall back to the outermost {...} block
    start, end = s.find("{"), s.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(s[start:end + 1])
        except Exception:
            return None
    return None


def _table_to_df(tbl: dict) -> Optional[pd.DataFrame]:
    """Turn one {columns, rows} object into a DataFrame, normalising row widths."""
    cols = tbl.get("columns") or []
    rows = tbl.get("rows") or []
    if not rows:
        return None
    width = len(cols) if cols else max((len(r) for r in rows), default=0)
    if width == 0:
        return None
    norm = []
    for r in rows:
        r = list(r)
        if len(r) < width:
            r = r + [None] * (width - len(r))
        elif len(r) > width:
            r = r[:width]
        norm.append(r)
    columns = cols if len(cols) == width else [f"col{i}" for i in range(width)]
    df = pd.DataFrame(norm, columns=columns)
    return df


# --------------------------------------------------------------------------- reconciliation (shared gate)

_NUM_RE = re.compile(r"^\(?\$?\s*-?[\d,]+(?:\.\d+)?\)?\s*%?$")


def _to_number(cell: object) -> Optional[float]:
    if cell is None:
        return None
    s = str(cell).strip()
    if s in ("", "—", "-", "–", "nan", "None", "null"):
        return None
    s = s.replace("$", "").replace(",", "").replace("%", "").replace("\n", " ").strip()
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()").strip()
    if not re.match(r"^-?\d+(\.\d+)?$", s):
        return None
    v = float(s)
    return -v if neg else v


def _first_numeric_col(df: pd.DataFrame) -> Optional[int]:
    best, best_count = None, 0
    for c in range(df.shape[1]):
        cnt = sum(_to_number(v) is not None for v in df.iloc[:, c])
        if cnt > best_count:
            best, best_count = c, cnt
    return best if best_count >= 3 else None


def _reconciles(df: pd.DataFrame, tol: float = 0.02) -> Optional[bool]:
    """Same subtotal gate as the rule-based version: True/False/None."""
    if df is None or df.empty or df.shape[1] < 2:
        return None
    col = _first_numeric_col(df)
    if col is None:
        return None
    label_col = 0 if col != 0 else 1
    parts, total = [], None
    for _, row in df.iterrows():
        label = str(row.iloc[label_col]).strip().lower()
        v = _to_number(row.iloc[col])
        if v is None:
            continue
        if "total" in label:          # covers 'total' and 'subtotal'
            total = v
        else:
            parts.append(v)
    if total is None or not parts:
        return None
    for cut in range(len(parts), 0, -1):
        if abs(sum(parts[:cut]) - total) <= max(abs(total) * tol, 1.0):
            return True
    return abs(sum(parts) - total) <= max(abs(total) * tol, 1.0)


# --------------------------------------------------------------------------- page candidate finder

def _candidate_pages(pdf_path: str) -> List[int]:
    """Same permissive pre-filter as the rule-based extractor."""
    import pdfplumber
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, pg in enumerate(pdf.pages, start=1):
            found = bool(pg.find_tables())
            if not found:
                try:
                    tt = pg.find_tables({
                        "vertical_strategy": "text",
                        "horizontal_strategy": "text",
                    })
                    found = any(
                        t.bbox and (t.bbox[2] - t.bbox[0]) > 200
                        and len(t.rows) >= 3 and len(t.columns) >= 3
                        for t in tt
                    )
                except Exception:
                    found = False
            if found:
                pages.append(i)
    return pages


# --------------------------------------------------------------------------- driver

def extract_tables_vision(
    pdf_path: str,
    client,
    model: str,
    pages: Optional[List[int]] = None,
    dpi: int = 150,
) -> List[Tuple[str, pd.DataFrame]]:
    """
    Extract tables from a PDF using a vision model.

    Parameters
    ----------
    pdf_path : str
    client   : OpenAI-style client (has .chat.completions.create)
    model    : model name string passed straight to the gateway
    pages    : optional list of 1-based page numbers. If None, auto-detect
               candidate pages (same pre-filter as the rule-based extractor).
    dpi      : render resolution; bump to 200 for very dense tables.

    Returns
    -------
    list of (label, dataframe). label encodes page, source, and gate result,
    e.g. "p57 :: vision :: reconciled".
    """
    if pages is None:
        pages = _candidate_pages(pdf_path)

    results: List[Tuple[str, pd.DataFrame]] = []

    for page in pages:
        try:
            png = _render_page_png(pdf_path, page, dpi=dpi)
            raw = _call_vision(client, model, png)
        except Exception as e:
            results.append((f"p{page} :: vision :: error({type(e).__name__})",
                            pd.DataFrame()))
            continue

        parsed = _extract_json(raw)
        if not parsed or "tables" not in parsed:
            results.append((f"p{page} :: vision :: unparseable", pd.DataFrame()))
            continue

        for tbl in parsed["tables"]:
            df = _table_to_df(tbl)
            if df is None or df.empty:
                continue
            g = _reconciles(df)
            tag = {True: "reconciled", False: "failed-gate", None: "unverified"}[g]
            title = (tbl.get("title") or "").strip()
            suffix = f" :: {title}" if title else ""
            results.append((f"p{page} :: vision :: {tag}{suffix}", df))

    return results


# --------------------------------------------------------------------------- demo / self-test

if __name__ == "__main__":
    import sys

    # A tiny fake client so the parsing + reconciliation path can be tested
    # WITHOUT a real gateway. It returns a canned JSON response that mimics a
    # vision model reading the page-57 balance-sheet table.
    class _FakeMessage:
        def __init__(self, content): self.content = content

    class _FakeChoice:
        def __init__(self, content): self.message = _FakeMessage(content)

    class _FakeResp:
        def __init__(self, content): self.choices = [_FakeChoice(content)]

    class _FakeCompletions:
        def create(self, model, messages, **kw):
            canned = {
                "tables": [{
                    "title": "Selected Consolidated balance sheets data",
                    "columns": ["December 31, (in millions)", "2025", "2024", "Change"],
                    "rows": [
                        ["Cash and due from banks", "$ 21,742", "$ 23,372", "(7)%"],
                        ["Deposits with banks", "321,596", "445,945", "(28)"],
                        ["Trading assets", "802,873", "637,784", "26"],
                        ["Total assets", "$ 4,424,900", "$ 4,002,814", "11%"],
                    ],
                }]
            }
            return _FakeResp(json.dumps(canned))

    class _FakeChat:
        def __init__(self): self.completions = _FakeCompletions()

    class _FakeClient:
        def __init__(self): self.chat = _FakeChat()

    path = sys.argv[1] if len(sys.argv) > 1 else "corp-10k-2025.pdf"
    out = extract_tables_vision(path, _FakeClient(), "fake-vision-model", pages=[57])
    for label, df in out:
        print("###", label)
        print(df.to_string(index=False))
