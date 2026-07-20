"""SQLite fact store + flat vector index (collapsed-tree style: one index over
all live facts of every type, no level routing).

Schema follows the plan:
  facts(id, text, embedding_id, type, confidence, created_at, valid_from,
        invalidated_at, stale, source_session, extraction_model, depth)
  provenance(child_id, parent_id)      -- DAG; observed facts have no parents
  entities(id, name, canonical_id)
  fact_entities(fact_id, entity_id)
  contradictions(fact_id, contradicted_by_fact_id, detected_at)

`stale` is soft invalidation: excluded from retrieval, kept for lazy
recomputation. `invalidated_at` is hard invalidation of the fact itself.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lcm_mem.encoder.embed import BaseEmbedder, l2_normalize

FACT_TYPES = ("observed", "derived", "world_bridge")


@dataclass
class Fact:
    id: int
    text: str
    type: str
    confidence: float
    depth: int
    stale: bool
    invalidated_at: float | None
    source_session: str | None


class VectorIndex:
    """Exact inner-product search (flat index). Ids are the store's fact ids;
    deletions are handled by the caller filtering on liveness, so the index
    itself is append-only.

    Backend is numpy by default: corpora here are small and search is exact
    either way, and faiss-cpu's bundled OpenMP aborts on macOS when torch is
    loaded in the same process. Set LCM_USE_FAISS=1 to opt in to FAISS."""

    def __init__(self, dim: int):
        import os

        self.dim = dim
        self.ids: list[int] = []
        self._faiss = None
        self._vecs: np.ndarray | None = None
        if os.environ.get("LCM_USE_FAISS") == "1":
            import faiss

            self._faiss = faiss.IndexFlatIP(dim)
        else:
            self._vecs = np.empty((0, dim), dtype=np.float32)

    def add(self, fact_id: int, vec: np.ndarray) -> None:
        v = l2_normalize(vec.astype(np.float32).reshape(1, -1))
        if self._faiss is not None:
            self._faiss.add(v)
        else:
            self._vecs = np.vstack([self._vecs, v])
        self.ids.append(fact_id)

    def search(self, query_vec: np.ndarray, k: int) -> list[tuple[int, float]]:
        if not self.ids:
            return []
        q = l2_normalize(query_vec.astype(np.float32).reshape(1, -1))
        k = min(k, len(self.ids))
        if self._faiss is not None:
            sims, idxs = self._faiss.search(q, k)
            return [
                (self.ids[i], float(s))
                for i, s in zip(idxs[0], sims[0])
                if i != -1
            ]
        sims = (self._vecs @ q[0])
        top = np.argsort(-sims)[:k]
        return [(self.ids[i], float(sims[i])) for i in top]


class MemoryStore:
    def __init__(self, db_path: str | Path, embedder: BaseEmbedder):
        if str(db_path) != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(db_path))
        self.db.row_factory = sqlite3.Row
        self.embedder = embedder
        self._embeddings: dict[int, np.ndarray] = {}
        self.index: VectorIndex | None = None
        self._init_schema()

    def _init_schema(self) -> None:
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY,
                text TEXT NOT NULL,
                type TEXT NOT NULL CHECK (type IN ('observed','derived','world_bridge')),
                confidence REAL NOT NULL,
                created_at REAL NOT NULL,
                valid_from REAL,
                invalidated_at REAL,
                stale INTEGER NOT NULL DEFAULT 0,
                source_session TEXT,
                extraction_model TEXT,
                depth INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS provenance (
                child_id INTEGER NOT NULL REFERENCES facts(id),
                parent_id INTEGER NOT NULL REFERENCES facts(id),
                PRIMARY KEY (child_id, parent_id)
            );
            CREATE INDEX IF NOT EXISTS idx_prov_parent ON provenance(parent_id);
            CREATE TABLE IF NOT EXISTS entities (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                canonical_id INTEGER REFERENCES entities(id)
            );
            CREATE TABLE IF NOT EXISTS fact_entities (
                fact_id INTEGER NOT NULL REFERENCES facts(id),
                entity_id INTEGER NOT NULL REFERENCES entities(id),
                PRIMARY KEY (fact_id, entity_id)
            );
            CREATE TABLE IF NOT EXISTS contradictions (
                fact_id INTEGER NOT NULL REFERENCES facts(id),
                contradicted_by_fact_id INTEGER NOT NULL REFERENCES facts(id),
                detected_at REAL NOT NULL
            );
            """
        )
        self.db.commit()

    # -- facts ------------------------------------------------------------

    def add_fact(
        self,
        text: str,
        type: str = "observed",
        confidence: float = 1.0,
        parents: list[int] | None = None,
        entities: list[str] | None = None,
        source_session: str | None = None,
        extraction_model: str | None = None,
        valid_from: float | None = None,
        embedding: np.ndarray | None = None,
    ) -> int:
        assert type in FACT_TYPES
        parents = parents or []
        depth = 0
        if parents:
            rows = self.db.execute(
                f"SELECT MAX(depth) FROM facts WHERE id IN ({','.join('?' * len(parents))})",
                parents,
            ).fetchone()
            depth = (rows[0] or 0) + 1
        cur = self.db.execute(
            """INSERT INTO facts
               (text, type, confidence, created_at, valid_from, source_session,
                extraction_model, depth)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (text, type, confidence, time.time(), valid_from, source_session,
             extraction_model, depth),
        )
        fact_id = cur.lastrowid
        for p in parents:
            self.db.execute(
                "INSERT OR IGNORE INTO provenance (child_id, parent_id) VALUES (?, ?)",
                (fact_id, p),
            )
        for name in entities or []:
            self.link_entity(fact_id, name)
        self.db.commit()

        vec = (
            embedding
            if embedding is not None
            else self.embedder.encode([text], kind="passage")[0]
        )
        self._embeddings[fact_id] = vec.astype(np.float32)
        if self.index is None:
            self.index = VectorIndex(dim=vec.shape[-1])
        self.index.add(fact_id, vec)
        return fact_id

    def get_fact(self, fact_id: int) -> Fact:
        r = self.db.execute("SELECT * FROM facts WHERE id = ?", (fact_id,)).fetchone()
        if r is None:
            raise KeyError(fact_id)
        return Fact(
            id=r["id"], text=r["text"], type=r["type"], confidence=r["confidence"],
            depth=r["depth"], stale=bool(r["stale"]),
            invalidated_at=r["invalidated_at"], source_session=r["source_session"],
        )

    def embedding(self, fact_id: int) -> np.ndarray:
        if fact_id not in self._embeddings:
            self._embeddings[fact_id] = self.embedder.encode(
                [self.get_fact(fact_id).text], kind="passage"
            )[0]
        return self._embeddings[fact_id]

    def is_live(self, fact_id: int) -> bool:
        r = self.db.execute(
            "SELECT invalidated_at IS NULL AND stale = 0 FROM facts WHERE id = ?",
            (fact_id,),
        ).fetchone()
        return bool(r and r[0])

    def live_fact_ids(self) -> list[int]:
        return [
            r[0]
            for r in self.db.execute(
                "SELECT id FROM facts WHERE invalidated_at IS NULL AND stale = 0"
            )
        ]

    def mark_invalidated(self, fact_id: int, by_fact_id: int | None = None) -> None:
        self.db.execute(
            "UPDATE facts SET invalidated_at = ? WHERE id = ? AND invalidated_at IS NULL",
            (time.time(), fact_id),
        )
        if by_fact_id is not None:
            self.db.execute(
                "INSERT INTO contradictions VALUES (?, ?, ?)",
                (fact_id, by_fact_id, time.time()),
            )
        self.db.commit()

    def mark_stale(self, fact_id: int, stale: bool = True) -> None:
        self.db.execute(
            "UPDATE facts SET stale = ? WHERE id = ?", (int(stale), fact_id)
        )
        self.db.commit()

    # -- provenance -------------------------------------------------------

    def parents(self, fact_id: int) -> list[int]:
        return [
            r[0]
            for r in self.db.execute(
                "SELECT parent_id FROM provenance WHERE child_id = ?", (fact_id,)
            )
        ]

    def children(self, fact_id: int) -> list[int]:
        return [
            r[0]
            for r in self.db.execute(
                "SELECT child_id FROM provenance WHERE parent_id = ?", (fact_id,)
            )
        ]

    # -- entities ---------------------------------------------------------

    def link_entity(self, fact_id: int, name: str) -> int:
        row = self.db.execute(
            "SELECT id FROM entities WHERE name = ?", (name,)
        ).fetchone()
        if row is None:
            cur = self.db.execute("INSERT INTO entities (name) VALUES (?)", (name,))
            entity_id = cur.lastrowid
        else:
            entity_id = row[0]
        self.db.execute(
            "INSERT OR IGNORE INTO fact_entities VALUES (?, ?)", (fact_id, entity_id)
        )
        return entity_id

    def canonical_entity_id(self, entity_id: int) -> int:
        row = self.db.execute(
            "SELECT canonical_id FROM entities WHERE id = ?", (entity_id,)
        ).fetchone()
        return row[0] if row and row[0] is not None else entity_id

    def set_canonical(self, entity_id: int, canonical_id: int) -> None:
        self.db.execute(
            "UPDATE entities SET canonical_id = ? WHERE id = ?",
            (canonical_id, entity_id),
        )
        self.db.commit()

    def facts_with_entities(self, names: list[str], live_only: bool = True) -> list[int]:
        """Facts sharing at least one canonical entity with `names`."""
        if not names:
            return []
        canon: set[int] = set()
        for name in names:
            row = self.db.execute(
                "SELECT id FROM entities WHERE name = ?", (name,)
            ).fetchone()
            if row:
                canon.add(self.canonical_entity_id(row[0]))
        if not canon:
            return []
        rows = self.db.execute(
            """SELECT DISTINCT fe.fact_id, e.id, e.canonical_id
               FROM fact_entities fe JOIN entities e ON fe.entity_id = e.id"""
        ).fetchall()
        hits = {
            r["fact_id"]
            for r in rows
            if (r["canonical_id"] or r["id"]) in canon
        }
        if live_only:
            hits = {f for f in hits if self.is_live(f)}
        return sorted(hits)

    def entities_of(self, fact_id: int) -> list[str]:
        return [
            r[0]
            for r in self.db.execute(
                """SELECT e.name FROM fact_entities fe
                   JOIN entities e ON fe.entity_id = e.id WHERE fe.fact_id = ?""",
                (fact_id,),
            )
        ]

    # -- search -----------------------------------------------------------

    def search(self, query: str, k: int = 20, live_only: bool = True) -> list[tuple[int, float]]:
        if self.index is None:
            return []
        qvec = self.embedder.encode([query], kind="query")[0]
        # over-fetch: dead facts remain in the append-only index
        hits = self.index.search(qvec, k * 4 if live_only else k)
        if live_only:
            hits = [(fid, s) for fid, s in hits if self.is_live(fid)]
        return hits[:k]
