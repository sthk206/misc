"""Local embedding with sentence-transformers on MPS, with a disk cache.

Embeddings are always computed locally (never via the gateway) so the encoder
can later be fine-tuned. Cache: one .npy per text, keyed by
sha1(model_rev | kind | text), sharded by the first two hex chars.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Iterable, Literal

import numpy as np

Kind = Literal["passage", "query"]

# e5 models require these prefixes; bge queries use an instruction prefix.
_E5_PREFIX = {"passage": "passage: ", "query": "query: "}
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def pick_device() -> str:
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def l2_normalize(x: np.ndarray, axis: int = -1) -> np.ndarray:
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.clip(norm, 1e-12, None)


class BaseEmbedder:
    name: str
    dim: int

    def encode(
        self, texts: list[str], kind: Kind = "passage", batch_size: int = 64
    ) -> np.ndarray:
        raise NotImplementedError


class STEmbedder(BaseEmbedder):
    """sentence-transformers embedder with per-text .npy disk cache."""

    def __init__(
        self,
        model_name: str = "intfloat/e5-base-v2",
        device: str | None = None,
        cache_dir: str | Path | None = "cache/emb",
    ):
        self.name = model_name
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._device = device
        self._model = None  # lazy: loading the model takes seconds

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.name, device=self._device or pick_device()
            )
        return self._model

    @property
    def dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def _prefix(self, text: str, kind: Kind) -> str:
        lname = self.name.lower()
        if "e5" in lname:
            return _E5_PREFIX[kind] + text
        if "bge" in lname and kind == "query":
            return _BGE_QUERY_PREFIX + text
        return text

    def _cache_path(self, text: str, kind: Kind) -> Path | None:
        if self.cache_dir is None:
            return None
        key = hashlib.sha1(f"{self.name}|{kind}|{text}".encode()).hexdigest()
        return self.cache_dir / key[:2] / f"{key}.npy"

    def encode(
        self, texts: list[str], kind: Kind = "passage", batch_size: int = 64
    ) -> np.ndarray:
        out: list[np.ndarray | None] = [None] * len(texts)
        to_compute: list[int] = []
        for i, t in enumerate(texts):
            p = self._cache_path(t, kind)
            if p is not None and p.exists():
                out[i] = np.load(p)
            else:
                to_compute.append(i)
        if to_compute:
            vecs = self.model.encode(
                [self._prefix(texts[i], kind) for i in to_compute],
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=len(to_compute) > 256,
            ).astype(np.float32)
            for j, i in enumerate(to_compute):
                out[i] = vecs[j]
                p = self._cache_path(texts[i], kind)
                if p is not None:
                    p.parent.mkdir(parents=True, exist_ok=True)
                    np.save(p, vecs[j])
        return np.stack(out).astype(np.float32)  # type: ignore[arg-type]


class HashingEmbedder(BaseEmbedder):
    """Deterministic pseudo-embeddings for offline tests. No semantics, but
    identical text -> identical vector, and vectors are unit-norm."""

    def __init__(self, dim: int = 64):
        self.name = f"hash-{dim}"
        self.dim = dim

    def encode(
        self, texts: list[str], kind: Kind = "passage", batch_size: int = 64
    ) -> np.ndarray:
        vecs = np.empty((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            seed = int.from_bytes(
                hashlib.sha256(t.encode()).digest()[:8], "little"
            )
            rng = np.random.default_rng(seed)
            vecs[i] = rng.standard_normal(self.dim)
        return l2_normalize(vecs)


def get_embedder(name: str = "intfloat/e5-base-v2", **kw) -> BaseEmbedder:
    if name.startswith("hash"):
        dim = int(name.split("-")[1]) if "-" in name else 64
        return HashingEmbedder(dim=dim)
    return STEmbedder(model_name=name, **kw)


def embed_jsonl_field(
    embedder: BaseEmbedder,
    rows: Iterable[dict],
    field: str,
    kind: Kind = "passage",
    batch_size: int = 64,
) -> np.ndarray:
    return embedder.encode([r[field] for r in rows], kind=kind, batch_size=batch_size)
