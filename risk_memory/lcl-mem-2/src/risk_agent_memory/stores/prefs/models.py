"""S2 preference store (Phase B).

Preferences live ONLY here. Reads inject the whole confirmed profile at session
start (small by construction — no retrieval, no embeddings). Writes:
  explicit  -> `prefs_set` tool, status=confirmed immediately, source=explicit.
  inferred  -> reflector PREF_CANDIDATE, status=candidate, surfaced to the
               manager at the next morning report, confirmed only on yes.
The raw table is the audit view; deletion is honored next session (revocation).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from risk_agent_memory.db import connect, dumps, loads, now
from risk_agent_memory.stores.prefs.registry import PrefRegistry


@dataclass
class Preference:
    manager_id: str
    key: str
    value: Any
    source: str      # explicit | inferred
    status: str      # confirmed | candidate
    created_at: float
    last_used_at: float | None
    updated_by: str
    evidence: str | None = None


class PrefsStore:
    def __init__(self, db_path: str | Path, registry: PrefRegistry):
        self.db = connect(db_path)
        self.registry = registry
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS preference (
                manager_id TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,          -- JSON
                source TEXT NOT NULL CHECK (source IN ('explicit','inferred')),
                status TEXT NOT NULL CHECK (status IN ('confirmed','candidate')),
                created_at REAL NOT NULL,
                last_used_at REAL,
                updated_by TEXT NOT NULL,
                evidence TEXT,
                PRIMARY KEY (manager_id, key, status)
            );
            """
        )
        self.db.commit()

    # -- writes -----------------------------------------------------------

    def set(
        self, manager_id: str, key: str, value: Any,
        source: str = "explicit", updated_by: str = "manager",
    ) -> None:
        """Explicit write path: validated, confirmed immediately."""
        self.registry.validate(key, value)
        self.db.execute(
            "DELETE FROM preference WHERE manager_id=? AND key=?", (manager_id, key)
        )
        self.db.execute(
            "INSERT INTO preference VALUES (?, ?, ?, ?, 'confirmed', ?, NULL, ?, NULL)",
            (manager_id, key, dumps(value), source, now(), updated_by),
        )
        self.db.commit()

    def propose_candidate(
        self, manager_id: str, key: str, value: Any, evidence: str | None = None
    ) -> None:
        """Inferred write path (reflector). Never silently applied: stays
        `candidate` until the manager confirms. A confirmed value for the same
        key is left untouched."""
        self.registry.validate(key, value)
        self.db.execute(
            "DELETE FROM preference WHERE manager_id=? AND key=? AND status='candidate'",
            (manager_id, key),
        )
        self.db.execute(
            "INSERT INTO preference VALUES (?, ?, ?, 'inferred', 'candidate', ?, NULL, 'reflector', ?)",
            (manager_id, key, dumps(value), now(), evidence),
        )
        self.db.commit()

    def confirm(self, manager_id: str, key: str, by: str = "manager") -> None:
        r = self.db.execute(
            "SELECT value FROM preference WHERE manager_id=? AND key=? AND status='candidate'",
            (manager_id, key),
        ).fetchone()
        if r is None:
            raise KeyError(f"no candidate for {manager_id}/{key}")
        self.set(manager_id, key, loads(r["value"]), source="inferred", updated_by=by)

    def reject_candidate(self, manager_id: str, key: str) -> None:
        self.db.execute(
            "DELETE FROM preference WHERE manager_id=? AND key=? AND status='candidate'",
            (manager_id, key),
        )
        self.db.commit()

    def delete(self, manager_id: str, key: str) -> None:
        """Revocation: the deleted pref must stop applying next session."""
        self.db.execute(
            "DELETE FROM preference WHERE manager_id=? AND key=?", (manager_id, key)
        )
        self.db.commit()

    # -- reads ------------------------------------------------------------

    def profile(self, manager_id: str) -> dict[str, Any]:
        """The whole confirmed profile (what gets injected)."""
        rows = self.db.execute(
            "SELECT key, value FROM preference WHERE manager_id=? AND status='confirmed'",
            (manager_id,),
        ).fetchall()
        self.db.execute(
            "UPDATE preference SET last_used_at=? WHERE manager_id=? AND status='confirmed'",
            (now(), manager_id),
        )
        self.db.commit()
        return {r["key"]: loads(r["value"]) for r in rows}

    def candidates(self, manager_id: str) -> list[Preference]:
        return [
            Preference(
                manager_id=r["manager_id"], key=r["key"], value=loads(r["value"]),
                source=r["source"], status=r["status"], created_at=r["created_at"],
                last_used_at=r["last_used_at"], updated_by=r["updated_by"],
                evidence=r["evidence"],
            )
            for r in self.db.execute(
                "SELECT * FROM preference WHERE manager_id=? AND status='candidate'",
                (manager_id,),
            )
        ]

    def all_rows(self, manager_id: str | None = None) -> list[Preference]:
        """The audit view."""
        q = "SELECT * FROM preference"
        args: tuple = ()
        if manager_id:
            q += " WHERE manager_id=?"
            args = (manager_id,)
        return [
            Preference(
                manager_id=r["manager_id"], key=r["key"], value=loads(r["value"]),
                source=r["source"], status=r["status"], created_at=r["created_at"],
                last_used_at=r["last_used_at"], updated_by=r["updated_by"],
                evidence=r["evidence"],
            )
            for r in self.db.execute(q + " ORDER BY manager_id, key", args)
        ]


def render_profile_block(profile: dict[str, Any], registry: PrefRegistry) -> str:
    """Markdown block injected at session start. Mechanical keys are labeled
    with their consumer so skills bind them as template variables."""
    if not profile:
        return ""
    lines = ["## Manager preferences (confirmed)"]
    for key in sorted(profile):
        consumer = registry.consumer_of(key)
        lines.append(f"- `{key}` = {dumps(profile[key])}  (consumer: {consumer})")
    return "\n".join(lines)


def render_candidate_prompts(store: PrefsStore, manager_id: str) -> str:
    """Surfaced at the top of the next morning report — confirm-or-reject,
    never silently applied."""
    cands = store.candidates(manager_id)
    if not cands:
        return ""
    lines = ["## Preference suggestions (need your confirmation)"]
    for c in cands:
        ev = f" (noticed: {c.evidence})" if c.evidence else ""
        lines.append(
            f"- I noticed you may prefer `{c.key}` = {dumps(c.value)}{ev} — make it default? "
            f"Reply `prefs confirm {c.key}` or `prefs reject {c.key}`."
        )
    return "\n".join(lines)
