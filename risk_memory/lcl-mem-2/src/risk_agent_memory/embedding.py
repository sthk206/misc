"""Local embeddings for the dedup gate (A.3), pattern canonicalization (C.3),
and insight-abstraction search (C.5). Never routed through an LLM API.

`HashingEmbedder` provides deterministic pseudo-embeddings so every gate is
unit-testable offline; identical text -> identical unit vector.
"""

from __future__ import annotations

import hashlib

import numpy as np


def l2_normalize(x: np.ndarray, axis: int = -1) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.clip(n, 1e-12, None)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


class BaseEmbedder:
    name: str
    dim: int

    def encode(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError


class STEmbedder(BaseEmbedder):
    def __init__(self, model_name: str = "intfloat/e5-base-v2", device: str | None = None):
        self.name = model_name
        self._device = device
        self._model = None

    @property
    def model(self):
        if self._model is None:
            import os

            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
            import torch
            from sentence_transformers import SentenceTransformer

            device = self._device or ("mps" if torch.backends.mps.is_available() else "cpu")
            self._model = SentenceTransformer(self.name, device=device)
        return self._model

    @property
    def dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str]) -> np.ndarray:
        prefix = "passage: " if "e5" in self.name.lower() else ""
        return self.model.encode(
            [prefix + t for t in texts], normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)


class HashingEmbedder(BaseEmbedder):
    def __init__(self, dim: int = 64):
        self.name = f"hash-{dim}"
        self.dim = dim

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.empty((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            seed = int.from_bytes(hashlib.sha256(t.encode()).digest()[:8], "little")
            out[i] = np.random.default_rng(seed).standard_normal(self.dim)
        return l2_normalize(out)


def get_embedder(name: str = "intfloat/e5-base-v2") -> BaseEmbedder:
    if name.startswith("hash"):
        return HashingEmbedder(int(name.split("-")[1]) if "-" in name else 64)
    return STEmbedder(name)
