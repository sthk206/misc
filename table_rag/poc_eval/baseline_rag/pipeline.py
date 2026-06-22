"""
System 1: Baseline RAG.

A conventional, table-agnostic pipeline:
  PDF page text -> RecursiveCharacterTextSplitter(1000/200) -> gateway embeddings
  -> FAISS top-k -> single-shot answer generation.

Tables are NOT specially handled; they appear only as whatever linearized text the
PDF yields inside the page chunks. Chunk size/overlap, embedding model, retrieval
mechanism and generation LLM are all shared with TableRAG via common/* and the
gateway, to keep the comparison fair.
"""

from __future__ import annotations

import json
import os
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from poc_eval.common import llm_gateway
from poc_eval.common.jsonutil import extract_json
from poc_eval.common.retrieval import VectorIndex

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PAGES_PATH = os.path.join(ROOT, "poc_eval", "data", "pages.json")

CHUNK_SIZE = 1000      # mirrors online_inference/tools/retriever.py
CHUNK_OVERLAP = 200
TOP_K = 5

SYSTEM_PROMPT = (
    "You are a financial analyst answering questions about a company's 10-K filing. "
    "Use ONLY the provided context excerpts. The context is plain text extracted from "
    "the filing's pages and may include tables rendered as running text. "
    "If a question requires a calculation, perform it from the values in the context. "
    "Respond with a JSON object: {\"answer\": <concise answer, include units>, "
    "\"value\": <the single key number as a plain number with no units/commas, or null>, "
    "\"pages\": [<PDF page numbers you used>]}. "
    "If the answer is not in the context, set answer to \"NOT FOUND\"."
)


class BaselineRAG:
    name = "baseline_rag"

    def __init__(self) -> None:
        with open(PAGES_PATH) as f:
            pages = json.load(f)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        chunks: list[dict[str, Any]] = []
        for pg in pages:
            for ci, piece in enumerate(splitter.split_text(pg["text"])):
                # Prefix with page lineage, matching the repo's "File name:" convention.
                chunks.append(
                    {
                        "chunk_id": f"p{pg['pdf_page']}_c{ci}",
                        "pdf_page": pg["pdf_page"],
                        "section": pg["section"],
                        "text": f"[PDF page {pg['pdf_page']}]\n{piece}",
                    }
                )
        self.chunks = chunks
        self.index = VectorIndex(chunks, text_key="text")

    def answer(self, question: str) -> dict[str, Any]:
        hits = self.index.search(question, k=TOP_K)
        context = "\n\n---\n\n".join(h["text"] for h in hits)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ]
        raw = llm_gateway.chat_text(messages)
        parsed = extract_json(raw)
        return {
            "system": self.name,
            "question": question,
            "answer": parsed.get("answer", raw),
            "value": parsed.get("value"),
            "cited_pages": parsed.get("pages", []),
            "retrieved_evidence": [
                {"chunk_id": h["chunk_id"], "pdf_page": h["pdf_page"], "score": h["_score"]}
                for h in hits
            ],
            "raw_response": raw,
        }
