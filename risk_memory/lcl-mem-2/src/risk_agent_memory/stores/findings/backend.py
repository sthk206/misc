"""C.1 temporal-fact backend.

Production backend is Graphiti over Neo4j (`graphiti_backend.py`). This module
defines the narrow protocol the rest of S3 depends on, plus an in-memory
implementation with the same temporal semantics (validity windows, closing on
contradiction) used by tests and the eval fixtures — the sidecar DAG (C.2)
never depends on Graphiti internals, only on fact UUID references.
"""

from __future__ import annotations

import uuid as uuidlib
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class TemporalFact:
    uuid: str
    subject: str
    predicate: str
    object: str
    valid_from: float
    valid_to: float | None = None       # None = still valid
    invalid_at: float | None = None     # set when contradicted/retracted
    episode_ids: list[str] = field(default_factory=list)

    def valid_at(self, t: float) -> bool:
        if self.invalid_at is not None:
            return False
        return self.valid_from <= t and (self.valid_to is None or t < self.valid_to)


class FactBackend(Protocol):
    def add_fact(
        self, subject: str, predicate: str, object: str,
        valid_from: float, episode_id: str | None = None,
    ) -> tuple[str, list[str]]:
        """Returns (new fact uuid, uuids of facts whose validity was closed)."""
        ...

    def get(self, fact_uuid: str) -> TemporalFact: ...

    def search(
        self, entities: list[str] | None = None, as_of: float | None = None,
        text: str | None = None, k: int = 20,
    ) -> list[TemporalFact]: ...

    def invalidate(self, fact_uuid: str, at: float) -> None: ...


class InMemoryFactBackend:
    """Graphiti-like temporal semantics: ingesting a fact with the same
    (subject, predicate) but a different object closes the previous fact's
    validity window (returns the closed uuids so invalidation can propagate
    into the insight DAG)."""

    def __init__(self) -> None:
        self.facts: dict[str, TemporalFact] = {}

    def add_fact(
        self, subject: str, predicate: str, object: str,
        valid_from: float, episode_id: str | None = None,
    ) -> tuple[str, list[str]]:
        closed: list[str] = []
        for f in self.facts.values():
            if (
                f.subject == subject and f.predicate == predicate
                and f.object != object and f.valid_to is None
                and f.invalid_at is None and f.valid_from <= valid_from
            ):
                f.valid_to = valid_from
                closed.append(f.uuid)
        fid = str(uuidlib.uuid4())
        self.facts[fid] = TemporalFact(
            uuid=fid, subject=subject, predicate=predicate, object=object,
            valid_from=valid_from,
            episode_ids=[episode_id] if episode_id else [],
        )
        return fid, closed

    def get(self, fact_uuid: str) -> TemporalFact:
        return self.facts[fact_uuid]

    def search(
        self, entities: list[str] | None = None, as_of: float | None = None,
        text: str | None = None, k: int = 20,
    ) -> list[TemporalFact]:
        out = []
        for f in self.facts.values():
            if entities and not (f.subject in entities or f.object in entities):
                continue
            if as_of is not None and not f.valid_at(as_of):
                continue
            if as_of is None and f.invalid_at is not None:
                continue
            if text and text.lower() not in f"{f.subject} {f.predicate} {f.object}".lower():
                continue
            out.append(f)
        out.sort(key=lambda f: -f.valid_from)
        return out[:k]

    def invalidate(self, fact_uuid: str, at: float) -> None:
        self.facts[fact_uuid].invalid_at = at
