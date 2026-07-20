"""Provenance DAG semantics: contradiction detection and invalidation
propagation.

On ingesting a new fact we check for contradiction against existing facts
sharing >= 1 entity (candidate set = entity overlap + embedding sim > 0.75),
classify pairs with ONE batched gateway call, and on `contradicts`/`updates`
invalidate the old fact and propagate: every derived descendant in the
provenance DAG is marked stale (soft-invalid, kept for lazy recomputation).
"""

from __future__ import annotations

from collections import deque
from typing import Iterable, Mapping

import numpy as np

from lcm_mem.llm import prompts
from lcm_mem.llm.gateway import CachedGateway
from lcm_mem.memory.store import MemoryStore

SIM_THRESHOLD = 0.75


def descendant_closure(
    children_of: Mapping[int, Iterable[int]], roots: Iterable[int]
) -> set[int]:
    """Pure-graph BFS: all strict descendants of `roots` (roots excluded).
    Kept dependency-free for property-based testing."""
    out: set[int] = set()
    queue = deque(roots)
    while queue:
        node = queue.popleft()
        for child in children_of.get(node, ()):
            if child not in out:
                out.add(child)
                queue.append(child)
    return out


def invalidate_fact(store: MemoryStore, fact_id: int, by_fact_id: int | None = None) -> set[int]:
    """Hard-invalidate `fact_id`, then mark all and only its provenance
    descendants stale. Returns the set of stale-marked descendant ids."""
    store.mark_invalidated(fact_id, by_fact_id)
    children_of = {fid: store.children(fid) for fid in _reachable(store, fact_id)}
    stale = descendant_closure(children_of, [fact_id])
    for fid in stale:
        store.mark_stale(fid, True)
    return stale


def _reachable(store: MemoryStore, root: int) -> set[int]:
    seen = {root}
    queue = deque([root])
    while queue:
        node = queue.popleft()
        for child in store.children(node):
            if child not in seen:
                seen.add(child)
                queue.append(child)
    return seen


def contradiction_candidates(
    store: MemoryStore, new_fact_id: int, sim_threshold: float = SIM_THRESHOLD
) -> list[int]:
    """Existing live facts sharing >= 1 entity with the new fact AND embedding
    similarity above threshold."""
    names = store.entities_of(new_fact_id)
    shared = [f for f in store.facts_with_entities(names) if f != new_fact_id]
    if not shared:
        return []
    new_vec = store.embedding(new_fact_id)
    out = []
    for fid in shared:
        sim = float(np.dot(new_vec, store.embedding(fid)))
        if sim > sim_threshold:
            out.append(fid)
    return out


def classify_pairs(
    gateway: CachedGateway,
    model: str,
    pairs: list[tuple[str, str]],
) -> list[str]:
    """One batched gateway call classifying (old, new) statement pairs as
    contradicts | updates | duplicates | unrelated."""
    if not pairs:
        return []
    numbered = "\n".join(
        f"{i + 1}. OLD: {old}\n   NEW: {new}" for i, (old, new) in enumerate(pairs)
    )
    labels = gateway.chat_json(
        [{"role": "user", "content": prompts.CONTRADICTION_CLASSIFY_V1.format(
            numbered_pairs=numbered)}],
        model=model,
    )
    valid = {"contradicts", "updates", "duplicates", "unrelated"}
    return [str(l).lower() if str(l).lower() in valid else "unrelated" for l in labels]


def check_and_propagate(
    store: MemoryStore,
    new_fact_id: int,
    gateway: CachedGateway,
    model: str,
    sim_threshold: float = SIM_THRESHOLD,
) -> dict:
    """Full ingestion-time contradiction pipeline for one new fact."""
    candidates = contradiction_candidates(store, new_fact_id, sim_threshold)
    if not candidates:
        return {"checked": 0, "invalidated": [], "stale": []}
    new_text = store.get_fact(new_fact_id).text
    labels = classify_pairs(
        gateway, model, [(store.get_fact(c).text, new_text) for c in candidates]
    )
    invalidated, stale_all = [], set()
    for fid, label in zip(candidates, labels):
        if label in ("contradicts", "updates"):
            stale_all |= invalidate_fact(store, fid, by_fact_id=new_fact_id)
            invalidated.append(fid)
    return {
        "checked": len(candidates),
        "invalidated": invalidated,
        "stale": sorted(stale_all),
    }
