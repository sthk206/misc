"""S1 ACE playbook store (Phase A).

Single-ownership rules enforced here:
- Entries are imperative directives <= 60 tokens, no case details; conclusions
  live in S3, the entry only carries a `justification_ptr` to them.
- Nothing becomes `active` without human approval (the audit requirement).
- ADD/MERGE/RETIRE deltas queue for review; only helpful/harmful counters and
  the ADD->INCR dedup conversion apply automatically.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from risk_agent_memory.config import CONFIG
from risk_agent_memory.db import connect, dumps, loads, now
from risk_agent_memory.embedding import BaseEmbedder

SCOPE_GLOBAL = "global"
STATUSES = ("candidate", "active", "retired")
DELTA_KINDS = ("ADD", "INCR", "MERGE", "RETIRE", "PREF_CANDIDATE")


def approx_tokens(text: str) -> int:
    """Cheap token estimate (chars/4) — used for the budget, not billing."""
    return max(1, (len(text) + 3) // 4)


def scope_specificity(scope: str) -> int:
    """manager:<id> (2) > mode:<m> (1) > global (0)."""
    if scope.startswith("manager:"):
        return 2
    if scope.startswith("mode:"):
        return 1
    return 0


def scope_matches(scope: str, manager: str, mode: str) -> bool:
    return scope in (SCOPE_GLOBAL, f"manager:{manager}", f"mode:{mode}")


@dataclass
class PlaybookEntry:
    id: int
    text: str
    scope: str
    status: str
    helpful_count: int
    harmful_count: int
    last_fired_at: float | None
    justification_ptr: str | None
    created_by: str
    created_at: float
    approved_by: str | None
    approved_at: float | None

    @property
    def score(self) -> int:
        return self.helpful_count - self.harmful_count


class EntryTooLong(ValueError):
    pass


class AceStore:
    def __init__(self, db_path: str | Path, embedder: BaseEmbedder):
        self.db = connect(db_path)
        self.embedder = embedder
        self._init_schema()

    def _init_schema(self) -> None:
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS playbook_entry (
                id INTEGER PRIMARY KEY,
                text TEXT NOT NULL,
                scope TEXT NOT NULL,
                status TEXT NOT NULL CHECK (status IN ('candidate','active','retired')),
                helpful_count INTEGER NOT NULL DEFAULT 0,
                harmful_count INTEGER NOT NULL DEFAULT 0,
                last_fired_at REAL,
                justification_ptr TEXT,
                created_by TEXT NOT NULL,
                created_at REAL NOT NULL,
                approved_by TEXT,
                approved_at REAL
            );
            CREATE TABLE IF NOT EXISTS delta_queue (
                id INTEGER PRIMARY KEY,
                kind TEXT NOT NULL,
                payload TEXT NOT NULL,          -- JSON
                session_ref TEXT,
                status TEXT NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending','approved','rejected','auto_applied')),
                created_at REAL NOT NULL,
                decided_by TEXT,
                decided_at REAL
            );
            CREATE TABLE IF NOT EXISTS incr_evidence (
                entry_id INTEGER NOT NULL REFERENCES playbook_entry(id),
                direction TEXT NOT NULL CHECK (direction IN ('helpful','harmful')),
                evidence_span TEXT,
                session_ref TEXT,
                created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS pruning_queue (
                entry_id INTEGER NOT NULL REFERENCES playbook_entry(id),
                reason TEXT NOT NULL,
                created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS notifications (
                id INTEGER PRIMARY KEY,
                kind TEXT NOT NULL,
                payload TEXT NOT NULL,
                created_at REAL NOT NULL,
                resolved INTEGER NOT NULL DEFAULT 0
            );
            """
        )
        self.db.commit()

    # -- entries ----------------------------------------------------------

    def _row_to_entry(self, r) -> PlaybookEntry:
        return PlaybookEntry(
            id=r["id"], text=r["text"], scope=r["scope"], status=r["status"],
            helpful_count=r["helpful_count"], harmful_count=r["harmful_count"],
            last_fired_at=r["last_fired_at"],
            justification_ptr=r["justification_ptr"], created_by=r["created_by"],
            created_at=r["created_at"], approved_by=r["approved_by"],
            approved_at=r["approved_at"],
        )

    def add_entry(
        self,
        text: str,
        scope: str = SCOPE_GLOBAL,
        status: str = "candidate",
        created_by: str = "reflector",
        justification_ptr: str | None = None,
        approved_by: str | None = None,
    ) -> int:
        if approx_tokens(text) > CONFIG.thresholds.ace_entry_max_tokens:
            raise EntryTooLong(
                f"directive exceeds {CONFIG.thresholds.ace_entry_max_tokens} tokens"
            )
        if status == "active" and approved_by is None:
            raise ValueError("nothing becomes active without approval")
        cur = self.db.execute(
            """INSERT INTO playbook_entry
               (text, scope, status, justification_ptr, created_by, created_at,
                approved_by, approved_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (text, scope, status, justification_ptr, created_by, now(),
             approved_by, now() if approved_by else None),
        )
        self.db.commit()
        return cur.lastrowid

    def get(self, entry_id: int) -> PlaybookEntry:
        r = self.db.execute(
            "SELECT * FROM playbook_entry WHERE id = ?", (entry_id,)
        ).fetchone()
        if r is None:
            raise KeyError(entry_id)
        return self._row_to_entry(r)

    def entries(self, status: str | None = None) -> list[PlaybookEntry]:
        q = "SELECT * FROM playbook_entry"
        args: tuple = ()
        if status:
            q += " WHERE status = ?"
            args = (status,)
        return [self._row_to_entry(r) for r in self.db.execute(q + " ORDER BY id", args)]

    def active_for(self, manager: str, mode: str) -> list[PlaybookEntry]:
        return [
            e for e in self.entries("active") if scope_matches(e.scope, manager, mode)
        ]

    def set_status(self, entry_id: int, status: str, by: str | None = None) -> None:
        assert status in STATUSES
        if status == "active":
            if by is None:
                raise ValueError("activation requires an approver")
            self.db.execute(
                "UPDATE playbook_entry SET status=?, approved_by=?, approved_at=? WHERE id=?",
                (status, by, now(), entry_id),
            )
        else:
            self.db.execute(
                "UPDATE playbook_entry SET status=? WHERE id=?", (status, entry_id)
            )
        self.db.commit()

    def mark_fired(self, entry_id: int) -> None:
        self.db.execute(
            "UPDATE playbook_entry SET last_fired_at=? WHERE id=?", (now(), entry_id)
        )
        self.db.commit()

    def incr(
        self,
        entry_id: int,
        direction: str,
        evidence_span: str | None = None,
        session_ref: str | None = None,
    ) -> None:
        assert direction in ("helpful", "harmful")
        col = f"{direction}_count"
        self.db.execute(
            f"UPDATE playbook_entry SET {col} = {col} + 1 WHERE id = ?", (entry_id,)
        )
        self.db.execute(
            "INSERT INTO incr_evidence VALUES (?, ?, ?, ?, ?)",
            (entry_id, direction, evidence_span, session_ref, now()),
        )
        self.db.commit()

    # -- delta queue (A.3) ------------------------------------------------

    def submit_delta(
        self, kind: str, payload: dict[str, Any], session_ref: str | None = None
    ) -> int:
        """Route a reflector delta. Counters (INCR) apply immediately; ADD runs
        the dedup gate (may convert to INCR); ADD/MERGE/RETIRE queue for human
        review. Returns the delta row id."""
        assert kind in DELTA_KINDS, kind
        if kind == "INCR":
            self.incr(
                int(payload["entry_id"]), payload.get("direction", "helpful"),
                payload.get("evidence_span"), session_ref,
            )
            return self._log_delta(kind, payload, session_ref, "auto_applied")

        if kind == "ADD":
            dup = self.find_duplicate(payload["text"], payload.get("scope", SCOPE_GLOBAL))
            if dup is not None:
                self.incr(dup, "helpful", payload.get("evidence_span"), session_ref)
                return self._log_delta(
                    "INCR",
                    {"entry_id": dup, "direction": "helpful",
                     "converted_from": "ADD", "original_text": payload["text"]},
                    session_ref, "auto_applied",
                )
        return self._log_delta(kind, payload, session_ref, "pending")

    def _log_delta(self, kind, payload, session_ref, status) -> int:
        cur = self.db.execute(
            """INSERT INTO delta_queue (kind, payload, session_ref, status, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (kind, dumps(payload), session_ref, status, now()),
        )
        self.db.commit()
        return cur.lastrowid

    def find_duplicate(self, text: str, scope: str) -> int | None:
        """Dedup gate: cosine vs existing entries in the same scope; above the
        pinned threshold, the ADD becomes an INCR on the existing entry."""
        in_scope = [
            e for e in self.entries()
            if e.scope == scope and e.status in ("candidate", "active")
        ]
        if not in_scope:
            return None
        vecs = self.embedder.encode([e.text for e in in_scope])
        v = self.embedder.encode([text])[0]
        sims = vecs @ v
        best = int(np.argmax(sims))
        if float(sims[best]) > CONFIG.thresholds.ace_dedup_cosine:
            return in_scope[best].id
        return None

    def pending_deltas(self) -> list[dict]:
        return [
            {"id": r["id"], "kind": r["kind"], "payload": loads(r["payload"]),
             "session_ref": r["session_ref"], "created_at": r["created_at"]}
            for r in self.db.execute(
                "SELECT * FROM delta_queue WHERE status='pending' ORDER BY id"
            )
        ]

    def decide_delta(
        self, delta_id: int, approve: bool, by: str,
        edited_payload: dict[str, Any] | None = None,
    ) -> int | None:
        """Apply a human decision. On approval, mutate the playbook. Returns the
        affected entry id (for ADD/MERGE) or None."""
        r = self.db.execute(
            "SELECT * FROM delta_queue WHERE id=? AND status='pending'", (delta_id,)
        ).fetchone()
        if r is None:
            raise KeyError(f"no pending delta {delta_id}")
        payload = edited_payload or loads(r["payload"])
        new_status = "approved" if approve else "rejected"
        self.db.execute(
            "UPDATE delta_queue SET status=?, decided_by=?, decided_at=?, payload=? WHERE id=?",
            (new_status, by, now(), dumps(payload), delta_id),
        )
        self.db.commit()
        if not approve:
            return None

        kind = r["kind"]
        if kind == "ADD":
            return self.add_entry(
                payload["text"], payload.get("scope", SCOPE_GLOBAL),
                status="active", created_by="reflector",
                justification_ptr=payload.get("justification_ptr"), approved_by=by,
            )
        if kind == "MERGE":
            ids = [int(i) for i in payload["entry_ids"]]
            merged_text = payload.get("text") or self.get(ids[0]).text
            sources = [self.get(i) for i in ids]
            new_id = self.add_entry(
                merged_text,
                scope=payload.get("scope", sources[0].scope),
                status="active", created_by="promotion",
                justification_ptr=payload.get("justification_ptr")
                or sources[0].justification_ptr,
                approved_by=by,
            )
            for e in sources:
                # carry the evidence counters into the merged entry
                self.db.execute(
                    """UPDATE playbook_entry SET
                       helpful_count = helpful_count + ?,
                       harmful_count = harmful_count + ? WHERE id = ?""",
                    (e.helpful_count, e.harmful_count, new_id),
                )
                self.set_status(e.id, "retired")
            self.db.commit()
            return new_id
        if kind == "RETIRE":
            self.set_status(int(payload["entry_id"]), "retired")
            return None
        raise ValueError(f"unexpected pending delta kind {kind}")

    # -- pruning / notifications -----------------------------------------

    def flag_for_pruning(self, entry_id: int, reason: str) -> None:
        self.db.execute(
            "INSERT INTO pruning_queue VALUES (?, ?, ?)", (entry_id, reason, now())
        )
        self.db.commit()

    def notify(self, kind: str, payload: dict[str, Any]) -> None:
        self.db.execute(
            "INSERT INTO notifications (kind, payload, created_at) VALUES (?, ?, ?)",
            (kind, dumps(payload), now()),
        )
        self.db.commit()

    def open_notifications(self) -> list[dict]:
        return [
            {"id": r["id"], "kind": r["kind"], "payload": loads(r["payload"])}
            for r in self.db.execute(
                "SELECT * FROM notifications WHERE resolved=0 ORDER BY id"
            )
        ]

    def entries_justified_by(self, pattern_ref: str) -> list[PlaybookEntry]:
        return [
            self._row_to_entry(r)
            for r in self.db.execute(
                "SELECT * FROM playbook_entry WHERE justification_ptr = ?",
                (pattern_ref,),
            )
        ]
