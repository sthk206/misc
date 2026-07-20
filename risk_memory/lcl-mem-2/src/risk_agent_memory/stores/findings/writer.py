"""C.4 insight write path — the validation gates are the point.

1. Schema-forced insight object (narrative / abstraction / claims / tags).
2. Abstraction validator (hard gate): regex reject if the abstraction contains
   tickers, currency pairs, desk names, client ids, or absolute dates
   (deny-lists from atom's reference data); then one LLM generality check.
   Fail -> one rewrite attempt -> else stored as `needs_review`, never
   silently valid.
3. Pattern canonicalization (C.3).
4. Parent capture: the PostToolUse hook logged every atom snapshot id and fact
   UUID the session touched; the writer selects the load-bearing subset.
   No insight commits with zero parents.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import yaml

from risk_agent_memory.stores.findings.dag import FindingsDag

# absolute dates: 2024-05-01, 01/05/2024, "May 3, 2024", "3 May 2024", bare years
_DATE_PATTERNS = [
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),
    re.compile(
        r"\b(?:January|February|March|April|May|June|July|August|September|"
        r"October|November|December)\s+\d{1,2}(?:,?\s+\d{4})?\b"
    ),
    re.compile(
        r"\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|"
        r"September|October|November|December)(?:\s+\d{4})?\b"
    ),
    re.compile(r"\b(19|20)\d{2}\b"),
]


@dataclass
class ValidationResult:
    ok: bool
    reasons: list[str] = field(default_factory=list)


class AbstractionValidator:
    def __init__(self, denylists: dict[str, list[str]]):
        terms = []
        for values in denylists.values():
            terms.extend(str(v) for v in values)
        self._term_res = [
            (t, re.compile(rf"(?<![A-Za-z0-9]){re.escape(t)}(?![A-Za-z0-9])", re.IGNORECASE))
            for t in terms
        ]

    @classmethod
    def load(cls, path: str | Path) -> "AbstractionValidator":
        with open(path) as f:
            return cls(yaml.safe_load(f))

    def validate(self, abstraction: str) -> ValidationResult:
        reasons = [
            f"contains denied term {term!r}"
            for term, rx in self._term_res
            if rx.search(abstraction)
        ]
        for rx in _DATE_PATTERNS:
            m = rx.search(abstraction)
            if m:
                reasons.append(f"contains absolute date {m.group(0)!r}")
                break
        return ValidationResult(ok=not reasons, reasons=reasons)


@dataclass
class InsightDraft:
    narrative: str
    abstraction: str
    claims: list[dict]
    parents: list[dict]
    entity_tags: list[str] = field(default_factory=list)
    entity_type_tags: list[str] = field(default_factory=list)
    event_class: str = "unclassified"
    severity: float = 0.0
    pattern_name: str | None = None
    pattern_description: str | None = None
    session_ref: str | None = None
    manager_id: str | None = None


# generality check + rewrite are LLM calls in production (subagent), injected as
# callables so the gate logic is testable offline
GeneralityCheck = Callable[[str], bool]
Rewrite = Callable[[str, list[str]], str]


def write_insight(
    dag: FindingsDag,
    draft: InsightDraft,
    validator: AbstractionValidator,
    generality_check: GeneralityCheck | None = None,
    rewrite: Rewrite | None = None,
) -> tuple[int, str]:
    """Run the full gated write. Returns (insight_id, final_status)."""

    def _passes(abstraction: str) -> tuple[bool, list[str]]:
        res = validator.validate(abstraction)
        if not res.ok:
            return False, res.reasons
        if generality_check is not None and not generality_check(abstraction):
            return False, ["LLM generality check failed"]
        return True, []

    abstraction = draft.abstraction
    ok, reasons = _passes(abstraction)
    if not ok and rewrite is not None:
        abstraction = rewrite(abstraction, reasons)   # one rewrite attempt
        ok, reasons = _passes(abstraction)
    status = "valid" if ok else "needs_review"

    insight_id = dag.add_insight(
        narrative=draft.narrative,
        abstraction=abstraction,
        claims=draft.claims,
        parents=draft.parents,          # NoParentsError if empty — by design
        entity_tags=draft.entity_tags,
        entity_type_tags=draft.entity_type_tags,
        event_class=draft.event_class,
        status=status,
        severity=draft.severity,
        session_ref=draft.session_ref,
        manager_id=draft.manager_id,
    )
    if draft.pattern_name and draft.pattern_description:
        dag.canonicalize_pattern(
            draft.pattern_name, draft.pattern_description, insight_id
        )
    return insight_id, status


INSIGHT_WRITER_PROMPT = """\
You are the insight writer. From the investigation transcript, produce ONE
JSON insight object:
{"narrative": "<entity-specific surface story>",
 "abstraction": "<generalized causal statement — NO tickers, currency pairs,
                 desk names, client ids, or absolute dates>",
 "claims": [{"text": str, "epistemic": "observed"|"inferred"|"world_knowledge",
             "conf": float}],
 "entity_tags": [...], "entity_type_tags": [...], "event_class": str,
 "severity": 0-10,
 "pattern_name": "<short pattern name>",
 "pattern_description": "<entity-free description of the recurring pattern>",
 "load_bearing_refs": [{"type": "zep_fact"|"atom_snapshot", "ref": "<id>"}]}

`load_bearing_refs` must be the subset of the session's logged snapshot/fact ids
actually load-bearing for the claims. Output ONLY the JSON object.
"""
