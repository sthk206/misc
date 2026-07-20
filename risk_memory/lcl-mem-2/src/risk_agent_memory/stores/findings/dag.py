"""C.2 insight DAG + C.3 pattern registry (sidecar tables, NOT inside the
temporal-KG backend — invalidation semantics need transactional control the
backend does not expose).

Insights are shared institutional memory across managers (S2 controls
presentation only). Parents reference backend fact UUIDs, atom snapshot ids,
or other insights; every insight must have >= 1 parent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from risk_agent_memory.config import CONFIG
from risk_agent_memory.db import connect, dumps, loads, now
from risk_agent_memory.embedding import BaseEmbedder

INSIGHT_STATUSES = (
    "valid", "flagged_stale", "superseded", "retracted", "needs_review"
)
PARENT_TYPES = ("zep_fact", "atom_snapshot", "insight")


class NoParentsError(ValueError):
    """No insight commits with zero parents (C.4 rule 4)."""


@dataclass
class Insight:
    id: int
    narrative: str
    abstraction: str
    claims: list[dict]           # [{text, epistemic: observed|inferred|world_knowledge, conf}]
    entity_tags: list[str]
    entity_type_tags: list[str]
    event_class: str
    pattern_ids: list[int]
    parents: list[dict]          # [{type, ref, as_of}]
    status: str
    depth: int
    severity: float
    created_at: float
    session_ref: str | None
    manager_id: str | None
    shared: bool = True
    stale_cause: str | None = None


@dataclass
class Pattern:
    id: int
    name: str
    description: str
    status: str                  # active | review
    live_instances: int
    instance_insight_ids: list[int] = field(default_factory=list)


class FindingsDag:
    def __init__(self, db_path: str | Path, embedder: BaseEmbedder):
        self.db = connect(db_path)
        self.embedder = embedder
        self._init_schema()

    def _init_schema(self) -> None:
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS insight (
                id INTEGER PRIMARY KEY,
                narrative TEXT NOT NULL,
                abstraction TEXT NOT NULL,
                claims TEXT NOT NULL,
                entity_tags TEXT NOT NULL,
                entity_type_tags TEXT NOT NULL,
                event_class TEXT NOT NULL,
                status TEXT NOT NULL CHECK (status IN
                    ('valid','flagged_stale','superseded','retracted','needs_review')),
                depth INTEGER NOT NULL,
                severity REAL NOT NULL DEFAULT 0,
                created_at REAL NOT NULL,
                session_ref TEXT,
                manager_id TEXT,
                shared INTEGER NOT NULL DEFAULT 1,
                stale_cause TEXT,
                abstraction_emb BLOB
            );
            CREATE TABLE IF NOT EXISTS insight_parent (
                insight_id INTEGER NOT NULL REFERENCES insight(id),
                ptype TEXT NOT NULL CHECK (ptype IN ('zep_fact','atom_snapshot','insight')),
                ref TEXT NOT NULL,
                as_of REAL,
                PRIMARY KEY (insight_id, ptype, ref)
            );
            CREATE INDEX IF NOT EXISTS idx_parent_ref ON insight_parent(ptype, ref);
            CREATE TABLE IF NOT EXISTS pattern (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT NOT NULL,
                embedding BLOB,
                status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active','review')),
                live_instances INTEGER NOT NULL DEFAULT 0,
                created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS pattern_instance (
                pattern_id INTEGER NOT NULL REFERENCES pattern(id),
                insight_id INTEGER NOT NULL REFERENCES insight(id),
                PRIMARY KEY (pattern_id, insight_id)
            );
            CREATE TABLE IF NOT EXISTS pattern_review_queue (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT NOT NULL,
                insight_id INTEGER,
                proposed_at REAL NOT NULL,
                resolved INTEGER NOT NULL DEFAULT 0
            );
            """
        )
        self.db.commit()

    # -- insights ---------------------------------------------------------

    def add_insight(
        self,
        narrative: str,
        abstraction: str,
        claims: list[dict],
        parents: list[dict],
        entity_tags: list[str] | None = None,
        entity_type_tags: list[str] | None = None,
        event_class: str = "unclassified",
        status: str = "valid",
        severity: float = 0.0,
        session_ref: str | None = None,
        manager_id: str | None = None,
    ) -> int:
        if not parents:
            raise NoParentsError("insight must reference at least one parent")
        for p in parents:
            if p.get("type") not in PARENT_TYPES:
                raise ValueError(f"bad parent type {p.get('type')!r}")
        assert status in INSIGHT_STATUSES
        depth = 1
        for p in parents:
            if p["type"] == "insight":
                depth = max(depth, self.get_insight(int(p["ref"])).depth + 1)
        emb = self.embedder.encode([abstraction])[0].astype(np.float32)
        cur = self.db.execute(
            """INSERT INTO insight
               (narrative, abstraction, claims, entity_tags, entity_type_tags,
                event_class, status, depth, severity, created_at, session_ref,
                manager_id, shared, abstraction_emb)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)""",
            (narrative, abstraction, dumps(claims), dumps(entity_tags or []),
             dumps(entity_type_tags or []), event_class, status, depth, severity,
             now(), session_ref, manager_id, emb.tobytes()),
        )
        iid = cur.lastrowid
        for p in parents:
            self.db.execute(
                "INSERT OR IGNORE INTO insight_parent VALUES (?, ?, ?, ?)",
                (iid, p["type"], str(p["ref"]), p.get("as_of")),
            )
        self.db.commit()
        return iid

    def get_insight(self, insight_id: int) -> Insight:
        r = self.db.execute("SELECT * FROM insight WHERE id=?", (insight_id,)).fetchone()
        if r is None:
            raise KeyError(insight_id)
        parents = [
            {"type": p["ptype"], "ref": p["ref"], "as_of": p["as_of"]}
            for p in self.db.execute(
                "SELECT * FROM insight_parent WHERE insight_id=?", (insight_id,)
            )
        ]
        pattern_ids = [
            p["pattern_id"]
            for p in self.db.execute(
                "SELECT pattern_id FROM pattern_instance WHERE insight_id=?",
                (insight_id,),
            )
        ]
        return Insight(
            id=r["id"], narrative=r["narrative"], abstraction=r["abstraction"],
            claims=loads(r["claims"]), entity_tags=loads(r["entity_tags"]),
            entity_type_tags=loads(r["entity_type_tags"]),
            event_class=r["event_class"], pattern_ids=pattern_ids,
            parents=parents, status=r["status"], depth=r["depth"],
            severity=r["severity"], created_at=r["created_at"],
            session_ref=r["session_ref"], manager_id=r["manager_id"],
            shared=bool(r["shared"]), stale_cause=r["stale_cause"],
        )

    def set_status(self, insight_id: int, status: str, cause: str | None = None) -> None:
        assert status in INSIGHT_STATUSES
        self.db.execute(
            "UPDATE insight SET status=?, stale_cause=COALESCE(?, stale_cause) WHERE id=?",
            (status, cause, insight_id),
        )
        self.db.commit()

    def insights_with_parent(self, ptype: str, ref: str) -> list[int]:
        return [
            r["insight_id"]
            for r in self.db.execute(
                "SELECT insight_id FROM insight_parent WHERE ptype=? AND ref=?",
                (ptype, str(ref)),
            )
        ]

    def search_abstractions(
        self, query: str, k: int = 10, include_flagged: bool = True
    ) -> list[tuple[Insight, float]]:
        """Embedding search over insight ABSTRACTIONS (not narratives).
        Flagged insights are returned with their flag, never silently dropped."""
        rows = self.db.execute(
            "SELECT id, abstraction_emb, status FROM insight WHERE status != 'retracted'"
        ).fetchall()
        if not rows:
            return []
        q = self.embedder.encode([query])[0]
        scored = []
        for r in rows:
            if not include_flagged and r["status"] != "valid":
                continue
            emb = np.frombuffer(r["abstraction_emb"], dtype=np.float32)
            scored.append((r["id"], float(emb @ q)))
        scored.sort(key=lambda t: -t[1])
        return [(self.get_insight(i), s) for i, s in scored[:k]]

    # -- patterns (C.3) ---------------------------------------------------

    def canonicalize_pattern(
        self, name: str, description: str, insight_id: int | None = None,
        threshold: float | None = None,
    ) -> tuple[int | None, str]:
        """Canonicalization gate on every proposal: cosine vs existing pattern
        descriptions. Above threshold -> link to existing (returns its id).
        Below -> human review queue; returns (None, 'review').
        Target registry size: dozens; past ~100 the thresholds are wrong."""
        threshold = threshold or CONFIG.thresholds.pattern_canon_cosine
        rows = self.db.execute(
            "SELECT id, embedding FROM pattern WHERE status='active'"
        ).fetchall()
        v = self.embedder.encode([description])[0].astype(np.float32)
        if rows:
            sims = [
                (r["id"], float(np.frombuffer(r["embedding"], dtype=np.float32) @ v))
                for r in rows
            ]
            best_id, best_sim = max(sims, key=lambda t: t[1])
            if best_sim > threshold:
                if insight_id is not None:
                    self.link_pattern_instance(best_id, insight_id)
                return best_id, "linked"
        self.db.execute(
            "INSERT INTO pattern_review_queue (name, description, insight_id, proposed_at)"
            " VALUES (?, ?, ?, ?)",
            (name, description, insight_id, now()),
        )
        self.db.commit()
        return None, "review"

    def approve_pattern(self, review_id: int) -> int:
        r = self.db.execute(
            "SELECT * FROM pattern_review_queue WHERE id=? AND resolved=0", (review_id,)
        ).fetchone()
        if r is None:
            raise KeyError(review_id)
        emb = self.embedder.encode([r["description"]])[0].astype(np.float32)
        cur = self.db.execute(
            "INSERT INTO pattern (name, description, embedding, status, created_at)"
            " VALUES (?, ?, ?, 'active', ?)",
            (r["name"], r["description"], emb.tobytes(), now()),
        )
        pid = cur.lastrowid
        if r["insight_id"] is not None:
            self.link_pattern_instance(pid, r["insight_id"])
        self.db.execute(
            "UPDATE pattern_review_queue SET resolved=1 WHERE id=?", (review_id,)
        )
        self.db.commit()
        return pid

    def link_pattern_instance(self, pattern_id: int, insight_id: int) -> None:
        cur = self.db.execute(
            "INSERT OR IGNORE INTO pattern_instance VALUES (?, ?)",
            (pattern_id, insight_id),
        )
        if cur.rowcount:
            self.db.execute(
                "UPDATE pattern SET live_instances = live_instances + 1 WHERE id=?",
                (pattern_id,),
            )
        self.db.commit()

    def decrement_pattern_live(self, pattern_id: int) -> None:
        self.db.execute(
            "UPDATE pattern SET live_instances = MAX(live_instances - 1, 0) WHERE id=?",
            (pattern_id,),
        )
        self.db.commit()

    def get_pattern(self, pattern_id: int) -> Pattern:
        r = self.db.execute("SELECT * FROM pattern WHERE id=?", (pattern_id,)).fetchone()
        if r is None:
            raise KeyError(pattern_id)
        instances = [
            x["insight_id"]
            for x in self.db.execute(
                "SELECT insight_id FROM pattern_instance WHERE pattern_id=?",
                (pattern_id,),
            )
        ]
        return Pattern(
            id=r["id"], name=r["name"], description=r["description"],
            status=r["status"], live_instances=r["live_instances"],
            instance_insight_ids=instances,
        )

    def patterns(self) -> list[Pattern]:
        return [
            self.get_pattern(r["id"])
            for r in self.db.execute("SELECT id FROM pattern ORDER BY id")
        ]

    def registry_size_ok(self) -> bool:
        n = self.db.execute("SELECT COUNT(*) FROM pattern").fetchone()[0]
        return n <= CONFIG.thresholds.pattern_registry_soft_cap

    def pattern_review_queue(self) -> list[dict]:
        return [
            dict(r)
            for r in self.db.execute(
                "SELECT * FROM pattern_review_queue WHERE resolved=0 ORDER BY id"
            )
        ]
