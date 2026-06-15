"""
Memory pruning POC
==================

A small, swappable pruning pass for comparing memory-system behaviour. Memory
stores (mem0, A-Mem) never refine what they write, so they grow forever with
exact repeats ("I like banana" x3) and many small related facts ("I like
banana", "I like apple"). This module prunes a store *after* ingestion so you
can compare benchmark results across strategies:

    none   -> baseline, no pruning
    dedup  -> collapse near-identical memories into one
    merge  -> combine each cluster of related memories into one consolidated line

Three layers, only the backend is system-specific:

  1. cluster_duplicates / MemoryRecord
        Pure clustering over (id, text, vector). No mem0, no A-Mem.

  2. MemoryBackend (protocol) + Mem0Backend
        The ~5 methods the orchestrator needs. Port to A-Mem by writing an
        AMemBackend with the same 5 methods — layers 1 and 3 stay unchanged.

  3. consolidate(backend, ...)
        Orchestrator: list -> embed -> cluster -> dedup/merge. Talks only to
        the backend, so it is fully system-agnostic.

This is a POC: cosine similarity is a heuristic, not a guarantee of duplication
(it can group negations / different dates / numbers). Use dry_run=True to eyeball
clusters and tune `threshold` before committing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

import numpy as np

logger = logging.getLogger(__name__)


# ===========================================================================
# Layer 1 — pure, system-agnostic clustering
# ===========================================================================


@dataclass
class MemoryRecord:
    id: str
    text: str
    vector: np.ndarray  # 1-D dense embedding


def cluster_duplicates(records: list[MemoryRecord], threshold: float = 0.85) -> list[list[int]]:
    """Group records whose embeddings are near-duplicates.

    Two records are linked when cosine similarity >= `threshold`; clusters are
    the connected components (single-link / union-find). Returns a list of
    clusters, each a list of indices into `records`; every record appears in
    exactly one cluster (singletons included).

    O(n^2) in time and memory — fine for benchmark-scale per-user sets.
    """
    n = len(records)
    if n == 0:
        return []
    if n == 1:
        return [[0]]

    V = np.stack([np.asarray(r.vector, dtype=np.float32) for r in records])
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Vn = V / norms
    sim = Vn @ Vn.T

    parent = list(range(n))

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:  # path compression
            parent[x], x = root, parent[x]
        return root

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, j in np.argwhere(np.triu(sim >= threshold, k=1)):
        union(int(i), int(j))

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def _representative(records: list[MemoryRecord], cluster: list[int]) -> int:
    """Index of the cluster member to keep: longest text, id as tie-break."""
    return max(cluster, key=lambda idx: (len(records[idx].text), records[idx].id))


# ===========================================================================
# Layer 2 — backend protocol + mem0 implementation
# ===========================================================================


class MemoryBackend(Protocol):
    """The only system-specific surface. Implement these 5 for any memory system."""

    def list_memories(self, user_id: Optional[str]) -> list[tuple[str, str]]:
        """Return [(memory_id, text), ...] for the user (or whole store if None)."""
        ...

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts for clustering."""
        ...

    def delete(self, mem_id: str) -> None:
        """Delete one memory by id."""
        ...

    def replace_text(self, mem_id: str, new_text: str) -> None:
        """Rewrite a memory's text in place (re-embed + persist). Merge only."""
        ...

    def llm_complete(self, system: str, user: str) -> str:
        """One-shot LLM completion. Merge only."""
        ...


class Mem0Backend:
    """Maps the protocol onto a mem0 `Memory` instance exposed by your adapter.

    `adapter` must expose `._mem` (the mem0 Memory). Uses public methods so the
    SQLite history stays consistent — no raw vector-store edits.
    """

    def __init__(self, adapter: Any):
        self._mem = adapter._mem

    def list_memories(self, user_id: Optional[str]) -> list[tuple[str, str]]:
        res = self._mem.get_all(user_id=user_id) if user_id is not None else self._mem.get_all()
        results = res.get("results", res) if isinstance(res, dict) else res
        out = []
        for r in results:
            text = r.get("memory", r.get("data", "")) or ""
            out.append((str(r.get("id")), text))
        return out

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._mem.embedding_model.embed_batch(texts, "search")

    def delete(self, mem_id: str) -> None:
        self._mem.delete(mem_id)

    def replace_text(self, mem_id: str, new_text: str) -> None:
        emb = self._mem.embedding_model.embed(new_text, "update")
        self._mem._update_memory(mem_id, new_text, {new_text: emb})

    def llm_complete(self, system: str, user: str) -> str:
        return self._mem.llm.generate_response(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
        )


# ===========================================================================
# Layer 3 — orchestrator
# ===========================================================================


_MERGE_SYSTEM = (
    "Combine the following memory statements into a single concise memory. "
    "Merge facts that are about the same thing; preserve every distinct fact; "
    "do not invent information. Respond with only the consolidated memory text, "
    "on a single line."
)


@dataclass
class ConsolidationReport:
    strategy: str = "dedup"
    n_before: int = 0
    n_after: int = 0
    n_clusters: int = 0  # clusters that had > 1 member
    dropped_ids: list = field(default_factory=list)
    dry_run: bool = False

    @property
    def n_removed(self) -> int:
        return len(self.dropped_ids)

    def __str__(self) -> str:
        mode = "DRY-RUN " if self.dry_run else ""
        return (
            f"{mode}{self.strategy}: {self.n_before} -> {self.n_after} memories "
            f"({self.n_removed} removed across {self.n_clusters} clusters)"
        )


def consolidate(
    backend: MemoryBackend,
    user_id: Optional[str] = None,
    strategy: str = "dedup",
    threshold: float = 0.85,
    dry_run: bool = False,
) -> ConsolidationReport:
    """Prune embedding-near-duplicate memories from `backend`.

    Args:
        backend:   A MemoryBackend (e.g. Mem0Backend(adapter)).
        user_id:   Restrict to one user, or None for the whole store.
        strategy:  "dedup" (keep longest, delete the rest) or
                   "merge"  (LLM-combine each cluster into one line, then delete the rest).
        threshold: Cosine similarity at/above which memories are grouped.
                   Lower groups related facts ("banana"+"apple"); higher only near-identical.
        dry_run:   Report and print clusters, but make no changes.

    Returns a ConsolidationReport.
    """
    if strategy not in ("dedup", "merge"):
        raise ValueError(f"unknown strategy: {strategy!r} (expected 'dedup' or 'merge')")

    pairs = backend.list_memories(user_id)
    if not pairs:
        return ConsolidationReport(strategy=strategy, dry_run=dry_run)

    texts = [t for _, t in pairs]
    vectors = backend.embed(texts)
    records = [MemoryRecord(id=i, text=t, vector=np.asarray(v, dtype=np.float32))
               for (i, t), v in zip(pairs, vectors)]

    clusters = cluster_duplicates(records, threshold=threshold)
    multi = [c for c in clusters if len(c) > 1]

    dropped_ids: list[str] = []
    for cluster in multi:
        rep = _representative(records, cluster)
        rep_id = records[rep].id
        others = [records[idx].id for idx in cluster if idx != rep]
        member_texts = [records[idx].text for idx in cluster]

        if dry_run:
            logger.info("cluster (keep %s): %s", rep_id, member_texts)
            dropped_ids.extend(others)
            continue

        if strategy == "merge" and len(set(member_texts)) > 1:
            # Only call the LLM when the texts actually differ.
            merged = backend.llm_complete(_MERGE_SYSTEM, "\n".join(member_texts)).strip()
            if merged:
                backend.replace_text(rep_id, merged.splitlines()[0].strip())

        for oid in others:
            backend.delete(oid)
        dropped_ids.extend(others)

    report = ConsolidationReport(
        strategy=strategy,
        n_before=len(records),
        n_after=len(records) - len(dropped_ids),
        n_clusters=len(multi),
        dropped_ids=dropped_ids,
        dry_run=dry_run,
    )
    logger.info("%s", report)
    return report
