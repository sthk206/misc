"""
Memory consolidation (Option B)
================================

Embedding-based near-duplicate pruning that lives **entirely outside** the
memory system. It reads vectors out of the store, clusters them, and deletes
redundant ones — mem0 never sees this logic.

Three layers, only the bottom two are mem0/Qdrant specific:

  1. cluster_duplicates / choose_keep_and_drop
        Pure functions over `(id, text, vector)`. No mem0, no Qdrant import.
        Reuse these unchanged for A-Mem or any other system.

  2. export_qdrant / delete_qdrant
        Glue that pulls vectors from / deletes ids in a mem0 Qdrant store,
        using the same native `client` you already use in save_state().

  3. consolidate
        Orchestrator: export -> cluster -> pick keepers -> delete the rest.
        Call it from Mem0Adapter (see module docstring at the bottom).

Caveat: deleting straight from the vector store leaves mem0's SQLite history
DB (and any graph store) holding stale rows. Search results are unaffected,
which is all the benchmark reads — but don't trust get_all()'s history fields
after consolidating, and don't use this with a graph-enabled config.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared record type
# ---------------------------------------------------------------------------


@dataclass
class MemoryRecord:
    """One memory, normalized across whatever system produced it."""

    id: str
    text: str
    vector: np.ndarray  # 1-D dense embedding


@dataclass
class ConsolidationReport:
    n_before: int = 0
    n_after: int = 0
    n_clusters: int = 0          # clusters with >1 member (i.e. that had dups)
    dropped_ids: list = field(default_factory=list)
    dry_run: bool = False

    @property
    def n_removed(self) -> int:
        return len(self.dropped_ids)

    def __str__(self) -> str:
        mode = "DRY-RUN " if self.dry_run else ""
        return (
            f"{mode}consolidation: {self.n_before} -> {self.n_after} memories "
            f"({self.n_removed} removed across {self.n_clusters} duplicate clusters)"
        )


# ===========================================================================
# Layer 1 — pure, system-agnostic clustering
# ===========================================================================


def cluster_duplicates(records: list[MemoryRecord], threshold: float = 0.92) -> list[list[int]]:
    """Group records whose embeddings are near-duplicates.

    Two records are linked when their cosine similarity >= `threshold`; clusters
    are the connected components of that graph (single-link / union-find). This
    means A~B and B~C puts A, B, C together even if A and C are below threshold.

    Returns a list of clusters, each a list of indices into `records`. Every
    record appears in exactly one cluster (singletons included).

    O(n^2) in time and memory — fine for benchmark-scale per-user sets
    (hundreds to a few thousand). For much larger sets, swap in a blocked /
    ANN-based linker behind this same signature.
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

    sim = Vn @ Vn.T  # cosine similarity, n x n

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

    # Link every above-threshold pair (upper triangle, excluding diagonal).
    pairs = np.argwhere(np.triu(sim >= threshold, k=1))
    for i, j in pairs:
        union(int(i), int(j))

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def choose_keep_and_drop(
    records: list[MemoryRecord],
    clusters: list[list[int]],
    strategy: str = "longest",
) -> tuple[list[str], list[str]]:
    """Within each cluster, keep one representative and drop the rest.

    strategy:
      "longest" — keep the record with the longest text (most information).
                  Deterministic tie-break by id.
      "central" — keep the medoid (highest summed cosine similarity to the
                  others in its cluster). Falls back to "longest" for singletons.

    Returns (keep_ids, drop_ids).
    """
    keep_ids: list[str] = []
    drop_ids: list[str] = []

    # Precompute normalized vectors only if needed.
    Vn = None
    if strategy == "central":
        V = np.stack([np.asarray(r.vector, dtype=np.float32) for r in records])
        norms = np.linalg.norm(V, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        Vn = V / norms

    for cluster in clusters:
        if len(cluster) == 1:
            keep_ids.append(records[cluster[0]].id)
            continue

        if strategy == "central" and Vn is not None:
            sub = Vn[cluster]
            centrality = (sub @ sub.T).sum(axis=1)  # sum sim to cluster-mates
            winner = cluster[int(np.argmax(centrality))]
        else:  # "longest"
            winner = max(cluster, key=lambda idx: (len(records[idx].text), records[idx].id))

        keep_ids.append(records[winner].id)
        drop_ids.extend(records[idx].id for idx in cluster if idx != winner)

    return keep_ids, drop_ids


# ===========================================================================
# Layer 2 — mem0 / Qdrant glue
# ===========================================================================


def _extract_dense(vector: Any) -> Optional[np.ndarray]:
    """Pull the dense embedding out of a scrolled Qdrant point.

    mem0 stores named vectors {"": dense, "bm25": sparse}, so `point.vector`
    comes back as a dict. Plain lists (unnamed config) are handled too.
    """
    if vector is None:
        return None
    if isinstance(vector, dict):
        dense = vector.get("")
        if dense is None:  # first non-sparse entry as a fallback
            for v in vector.values():
                if isinstance(v, (list, np.ndarray)):
                    dense = v
                    break
        if dense is None:
            return None
        return np.asarray(dense, dtype=np.float32)
    return np.asarray(vector, dtype=np.float32)


def export_qdrant(mem: Any, user_id: Optional[str] = None, page: int = 256) -> list[MemoryRecord]:
    """Scroll every point (with vectors) out of a mem0 Qdrant store.

    `mem` is the mem0 Memory instance (adapter._mem). Uses the native client
    exactly like save_state, but with with_vectors=True so we get embeddings.
    """
    store = mem.vector_store
    client = store.client
    collection = store.collection_name

    scroll_filter = None
    if user_id is not None:
        # Built lazily so the pure layer never imports qdrant.
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        scroll_filter = Filter(
            must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
        )

    records: list[MemoryRecord] = []
    skipped = 0
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            scroll_filter=scroll_filter,
            limit=page,
            with_payload=True,
            with_vectors=True,
            offset=offset,
        )
        for p in points:
            dense = _extract_dense(p.vector)
            if dense is None:
                skipped += 1
                continue
            payload = p.payload or {}
            text = payload.get("data", payload.get("memory", "")) or ""
            records.append(MemoryRecord(id=str(p.id), text=text, vector=dense))
        if offset is None:
            break

    if skipped:
        logger.warning("export_qdrant: %d point(s) had no dense vector and were skipped", skipped)
    return records


def delete_qdrant(mem: Any, ids: list[str]) -> None:
    """Delete points by id from a mem0 Qdrant store (batch, native client)."""
    if not ids:
        return
    from qdrant_client.models import PointIdsList

    store = mem.vector_store
    store.client.delete(
        collection_name=store.collection_name,
        points_selector=PointIdsList(points=ids),
    )


# ===========================================================================
# Layer 3 — orchestrator
# ===========================================================================


def consolidate(
    adapter: Any,
    user_id: Optional[str] = None,
    threshold: float = 0.92,
    strategy: str = "longest",
    dry_run: bool = False,
) -> ConsolidationReport:
    """Prune embedding-near-duplicate memories from `adapter`'s store.

    Args:
        adapter:   Your Mem0Adapter (must expose `._mem`, the mem0 Memory).
        user_id:   Restrict to one user, or None to consolidate the whole store.
        threshold: Cosine similarity at/above which two memories are duplicates.
        strategy:  "longest" (keep most text) or "central" (keep the medoid).
        dry_run:   Compute and report, but don't delete anything.

    Returns a ConsolidationReport.
    """
    mem = adapter._mem

    records = export_qdrant(mem, user_id=user_id)
    clusters = cluster_duplicates(records, threshold=threshold)
    keep_ids, drop_ids = choose_keep_and_drop(records, clusters, strategy=strategy)

    report = ConsolidationReport(
        n_before=len(records),
        n_after=len(records) - len(drop_ids),
        n_clusters=sum(1 for c in clusters if len(c) > 1),
        dropped_ids=drop_ids,
        dry_run=dry_run,
    )

    if not dry_run and drop_ids:
        delete_qdrant(mem, drop_ids)

    logger.info("%s", report)
    return report
