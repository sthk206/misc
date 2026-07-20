"""LLM extraction pipeline: session text -> observed facts + entities.

One gateway call per session/message batch with a fixed extraction prompt
producing JSON. Facts stored as `observed`, confidence 1.0, depth 0.
Entity canonicalization: embed entity names, cluster by cosine > threshold,
LLM tie-break for ambiguous merges (cached).
"""

from __future__ import annotations

import numpy as np

from lcm_mem.llm import prompts
from lcm_mem.llm.gateway import CachedGateway
from lcm_mem.memory.provenance import check_and_propagate
from lcm_mem.memory.store import MemoryStore

CANON_HI = 0.92   # above: auto-merge
CANON_LO = 0.80   # between LO and HI: LLM tie-break; below: distinct


def ingest_text(
    store: MemoryStore,
    text: str,
    gateway: CachedGateway,
    model: str,
    session_id: str,
    check_contradictions: bool = True,
) -> list[int]:
    """Extract facts from one session/message batch and store them."""
    items = gateway.chat_json(
        [{"role": "user", "content": prompts.FACT_EXTRACTION_V1.format(text=text)}],
        model=model,
    )
    fact_ids = []
    for it in items:
        fid = store.add_fact(
            text=str(it["fact"]),
            type="observed",
            confidence=1.0,
            entities=[str(e) for e in it.get("entities", [])],
            source_session=session_id,
            extraction_model=model,
        )
        fact_ids.append(fid)
        if check_contradictions:
            check_and_propagate(store, fid, gateway, model)
    return fact_ids


def canonicalize_entities(
    store: MemoryStore,
    gateway: CachedGateway | None = None,
    model: str | None = None,
    hi: float = CANON_HI,
    lo: float = CANON_LO,
) -> int:
    """Cluster entity names by embedding cosine; ambiguous pairs get an LLM
    tie-break. Returns the number of merges applied. Greedy: each entity is
    merged into the earliest matching canonical representative."""
    rows = store.db.execute("SELECT id, name FROM entities ORDER BY id").fetchall()
    if len(rows) < 2:
        return 0
    names = [r["name"] for r in rows]
    vecs = store.embedder.encode(names, kind="passage")
    merges = 0
    canonical_reps: list[int] = []  # indices into rows
    for i in range(len(rows)):
        merged = False
        for j in canonical_reps:
            sim = float(np.dot(vecs[i], vecs[j]))
            if sim >= hi:
                same = True
            elif sim >= lo and gateway is not None and model is not None:
                ans = gateway.chat(
                    [{"role": "user", "content": prompts.ENTITY_MERGE_V1.format(
                        name_a=names[j], name_b=names[i])}],
                    model=model,
                ).strip().lower()
                same = ans.startswith("yes")
            else:
                same = False
            if same:
                store.set_canonical(rows[i]["id"], rows[j]["id"])
                merges += 1
                merged = True
                break
        if not merged:
            canonical_reps.append(i)
    return merges
