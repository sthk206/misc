"""
Shared dense retriever used by BOTH systems, so retrieval is identical and the
comparison stays fair. Mirrors the repo's design (FAISS IndexFlatIP) but embeds via
the gateway instead of local bge-m3.

Cosine similarity is obtained by L2-normalizing vectors and using inner product.
"""

from __future__ import annotations

from typing import Any, Optional

import faiss
import numpy as np

from poc_eval.common import llm_gateway


def _normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class VectorIndex:
    """A tiny FAISS-backed semantic index over a list of documents."""

    def __init__(self, docs: list[dict[str, Any]], text_key: str = "text"):
        """`docs` is a list of dicts; `text_key` selects the field to embed.
        The full dict is retained and returned on search so callers keep metadata."""
        self.docs = docs
        self.text_key = text_key
        self.index: Optional[faiss.Index] = None
        if docs:
            self._build()

    def _build(self) -> None:
        texts = [d[self.text_key] for d in self.docs]
        vecs = np.asarray(llm_gateway.embed(texts), dtype="float32")
        vecs = _normalize(vecs)
        self.dim = vecs.shape[1]
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(vecs)

    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        if not self.index or not self.docs:
            return []
        qv = np.asarray(llm_gateway.embed([query]), dtype="float32")
        qv = _normalize(qv)
        k = min(k, len(self.docs))
        scores, idxs = self.index.search(qv, k)
        results = []
        for score, i in zip(scores[0], idxs[0]):
            if i < 0:
                continue
            hit = dict(self.docs[i])
            hit["_score"] = float(score)
            results.append(hit)
        return results
