"""6.4 synthetic incident corpus generator with planted structure.

Produces N days of mock-atom feeds + news with:
- cross-incident ANALOG PAIRS: same causal pattern, different pair/desk, months
  apart (e.g. orphaned hedge in USDJPY early, EURUSD much later);
- DISTRACTORS: textually similar but causally different incidents (hedge closed
  the same day as expiry — benign; the trap flat retrieval falls into);
- RESTATEMENTS: cancel/correct events that retroactively change an incident's
  basis (the invalidation-correctness probe);
- scheduled events (expiries known in advance).

Ground truth is emitted by the generator; ingestion writes structured temporal
facts into the backend and gated insights into the DAG, so the retrieval /
invalidation metrics run fully offline (no LLM).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from risk_agent_memory.stores.findings.backend import InMemoryFactBackend
from risk_agent_memory.stores.findings.dag import FindingsDag
from risk_agent_memory.stores.findings.invalidation import on_fact_closed
from risk_agent_memory.stores.findings.writer import (
    AbstractionValidator, InsightDraft, write_insight,
)

ORPHANED_HEDGE_ABSTRACTION = (
    "A hedge position remains open after its underlying option expires, "
    "leaving unhedged directional exposure that goes unnoticed until risk "
    "figures move."
)
ORPHANED_HEDGE_PATTERN = (
    "orphaned-hedge",
    "A hedge outlives the position it was hedging, converting a hedged book "
    "into an outright directional exposure.",
)
BENIGN_UNWIND_ABSTRACTION = (
    "A hedge position is closed on the same day its underlying option expires, "
    "so no residual exposure remains."
)


@dataclass
class Incident:
    incident_id: str
    kind: str            # orphaned_hedge | benign_unwind
    pair: str
    desk: str
    day: int
    option_trade: str
    hedge_trade: str
    narrative: str = ""
    insight_id: int | None = None
    fact_uuids: list[str] = field(default_factory=list)


@dataclass
class GroundTruth:
    analog_pairs: list[tuple[str, str]] = field(default_factory=list)
    distractors: list[str] = field(default_factory=list)
    restated_incidents: list[str] = field(default_factory=list)
    incidents: dict[str, Incident] = field(default_factory=dict)


def _plant_incident(
    backend: InMemoryFactBackend,
    incident_id: str,
    kind: str,
    pair: str,
    desk: str,
    day: int,
) -> Incident:
    opt, hedge = f"OPT-{incident_id}", f"HDG-{incident_id}"
    inc = Incident(
        incident_id=incident_id, kind=kind, pair=pair, desk=desk, day=day,
        option_trade=opt, hedge_trade=hedge,
    )
    u1, _ = backend.add_fact(opt, "is_option_on", pair, valid_from=day - 20,
                             episode_id=f"feed:d{day - 20}")
    u2, _ = backend.add_fact(hedge, "hedges", opt, valid_from=day - 20,
                             episode_id=f"feed:d{day - 20}")
    u3, _ = backend.add_fact(opt, "status", "expired", valid_from=day,
                             episode_id=f"feed:d{day}")
    inc.fact_uuids = [u1, u2, u3]
    if kind == "orphaned_hedge":
        u4, _ = backend.add_fact(hedge, "status", "open", valid_from=day,
                                 episode_id=f"feed:d{day}")
        inc.fact_uuids.append(u4)
        inc.narrative = (
            f"{desk}: hedge {hedge} on {pair} remained open after option {opt} "
            f"expired on day {day}, leaving outright {pair} exposure."
        )
    else:
        u4, _ = backend.add_fact(hedge, "status", "closed", valid_from=day,
                                 episode_id=f"feed:d{day}")
        inc.fact_uuids.append(u4)
        inc.narrative = (
            f"{desk}: hedge {hedge} on {pair} was closed the same day option "
            f"{opt} expired on day {day}; no residual exposure."
        )
    return inc


def build_corpus(
    backend: InMemoryFactBackend,
    dag: FindingsDag,
    validator: AbstractionValidator,
    seed: int = 0,
) -> GroundTruth:
    gt = GroundTruth()

    plants = [
        ("INC-A", "orphaned_hedge", "USDJPY", "FX Options Desk", 25),
        ("INC-B", "orphaned_hedge", "EURUSD", "G10 Spot Desk", 115),
        ("INC-D", "benign_unwind", "GBPUSD", "FX Options Desk", 60),
    ]
    for incident_id, kind, pair, desk, day in plants:
        inc = _plant_incident(backend, incident_id, kind, pair, desk, day)
        gt.incidents[incident_id] = inc

        abstraction = (
            ORPHANED_HEDGE_ABSTRACTION if kind == "orphaned_hedge"
            else BENIGN_UNWIND_ABSTRACTION
        )
        pattern = ORPHANED_HEDGE_PATTERN if kind == "orphaned_hedge" else None
        draft = InsightDraft(
            narrative=inc.narrative,
            abstraction=abstraction,
            claims=[
                {"text": f"option {inc.option_trade} expired on day {day}",
                 "epistemic": "observed", "conf": 1.0},
                {"text": inc.narrative, "epistemic": "inferred", "conf": 0.9},
            ],
            parents=[{"type": "zep_fact", "ref": u, "as_of": day}
                     for u in inc.fact_uuids],
            entity_tags=[pair, desk, inc.option_trade, inc.hedge_trade],
            entity_type_tags=["CurrencyPair", "Desk", "OptionTrade", "HedgePosition"],
            event_class="expiry_risk",
            severity=7.0 if kind == "orphaned_hedge" else 1.0,
            pattern_name=pattern[0] if pattern else None,
            pattern_description=pattern[1] if pattern else None,
            session_ref=f"gen:{incident_id}",
        )
        inc.insight_id, _status = write_insight(dag, draft, validator)
        # the FIRST orphaned-hedge proposal lands in the review queue (nothing
        # to canonicalize against); approve it as the human would, immediately,
        # so the later analog instance links to it instead of minting a twin
        for row in dag.pattern_review_queue():
            dag.approve_pattern(row["id"])

    gt.analog_pairs.append(("INC-A", "INC-B"))
    gt.distractors.append("INC-D")
    return gt


def apply_restatement(
    backend: InMemoryFactBackend,
    dag: FindingsDag,
    gt: GroundTruth,
    incident_id: str = "INC-A",
    ace=None,
) -> list[str]:
    """Atom cancel/correct feed: the 'expired' status of the incident's option
    is restated (trade was actually cancelled earlier), closing the old fact —
    the invalidation trigger (C.6 b). Returns closed fact uuids."""
    inc = gt.incidents[incident_id]
    _, closed = backend.add_fact(
        inc.option_trade, "status", "cancelled_pre_expiry",
        valid_from=inc.day + 1, episode_id=f"restatement:{incident_id}",
    )
    gt.restated_incidents.append(incident_id)
    on_fact_closed(dag, closed, cause=f"atom restatement of {inc.option_trade}", ace=ace)
    return closed
