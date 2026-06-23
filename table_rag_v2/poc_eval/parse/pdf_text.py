"""
Extract per-page document text (the D / document-text modality).

This is shared by BOTH parser versions: the surrounding narrative text of the 10-K
is identical regardless of how well we parse the tables. Output is one JSON file per
PDF page in the `doc/` dir, in the key/value shape the repo's
`MixedDocRetriever.load_hybrid_dataset` expects (it concatenates ``f"{key} {item}\n"``
over the dict). The filename encodes the pdf page so retrieved chunks carry a citation.

Page convention matches benchmark_questions.json: pdf_pages are 1-based physical
pages as pdfplumber indexes them; printed_page = pdf_page - 2.
"""
from __future__ import annotations

import json
import os

import pdfplumber
from tqdm import tqdm


def extract_doc_text(pdf_path: str, out_dir: str) -> int:
    os.makedirs(out_dir, exist_ok=True)
    count = 0
    with pdfplumber.open(pdf_path) as pdf:
        for idx, page in enumerate(tqdm(pdf.pages, desc="doc-text")):
            pdf_page = idx + 1
            text = page.extract_text() or ""
            if not text.strip():
                continue
            record = {
                "pdf_page": pdf_page,
                "printed_page": pdf_page - 2,
                "text": text,
            }
            fname = f"page_{pdf_page:04d}.json"
            with open(os.path.join(out_dir, fname), "w", encoding="utf-8") as fout:
                json.dump(record, fout, ensure_ascii=False)
            count += 1
    return count


if __name__ == "__main__":
    import sys

    from poc_eval import config

    n = extract_doc_text(config.SOURCE_PDF, config.doc_dir(sys.argv[1] if len(sys.argv) > 1 else "auto"))
    print(f"wrote {n} page docs")
