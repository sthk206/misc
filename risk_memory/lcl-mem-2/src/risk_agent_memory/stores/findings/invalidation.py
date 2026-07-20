"""C.6 invalidation propagation (transactional, runs on trigger).

Triggers: (a) the backend closes/contradicts a fact's validity edge,
(b) atom restatement events (cancel/correct feed) mapped to fact
contradictions, (c) manual retraction.

Steps:
1. Insights with the invalidated ref in parents[] -> flagged_stale + cause.
2. Recurse UPWARD through insight->insight edges.
3. Decrement affected patterns' live instance counts.
4. Any ACE rule whose justification pattern dropped below its promotion
   evidence bar -> status candidate again + review-CLI notification.
5. Retrieval NEVER silently drops flagged insights (enforced in retrieval.py).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from risk_agent_memory.config import CONFIG
from risk_agent_memory.stores.ace.models import AceStore
from risk_agent_memory.stores.findings.dag import FindingsDag


@dataclass
class PropagationReport:
    flagged_insights: list[int] = field(default_factory=list)
    affected_patterns: list[int] = field(default_factory=list)
    weakened_ace_entries: list[int] = field(default_factory=list)


def propagate_invalidation(
    dag: FindingsDag,
    ref_type: str,             # zep_fact | atom_snapshot | insight
    ref: str,
    cause: str,
    ace: AceStore | None = None,
) -> PropagationReport:
    report = PropagationReport()

    seeds = dag.insights_with_parent(ref_type, str(ref))
    queue = deque(seeds)
    seen: set[int] = set()
    while queue:
        iid = queue.popleft()
        if iid in seen:
            continue
        seen.add(iid)
        ins = dag.get_insight(iid)
        if ins.status in ("retracted",):
            continue
        if ins.status != "flagged_stale":
            dag.set_status(iid, "flagged_stale", cause=cause)
            report.flagged_insights.append(iid)
        # step 2: recurse upward — insights that cite THIS insight as parent
        queue.extend(dag.insights_with_parent("insight", str(iid)))

    # step 3: pattern live counts
    touched_patterns: set[int] = set()
    for iid in report.flagged_insights:
        for pid in dag.get_insight(iid).pattern_ids:
            dag.decrement_pattern_live(pid)
            touched_patterns.add(pid)
    report.affected_patterns = sorted(touched_patterns)

    # step 4: ACE rules justified by weakened patterns
    if ace is not None:
        bar = CONFIG.thresholds.promotion_min_instances
        for pid in report.affected_patterns:
            p = dag.get_pattern(pid)
            if p.live_instances >= bar:
                continue
            for entry in ace.entries_justified_by(f"pattern:{pid}"):
                if entry.status != "active":
                    continue
                ace.set_status(entry.id, "candidate")
                ace.notify(
                    "evidence_weakened",
                    {"entry_id": entry.id, "pattern_id": pid, "cause": cause,
                     "message": "evidence weakened: reconfirm or retire"},
                )
                report.weakened_ace_entries.append(entry.id)
    return report


def on_fact_closed(
    dag: FindingsDag, closed_fact_uuids: list[str], cause: str,
    ace: AceStore | None = None,
) -> list[PropagationReport]:
    """Trigger (a)/(b): backend closed validity windows during ingestion."""
    return [
        propagate_invalidation(dag, "zep_fact", u, cause, ace)
        for u in closed_fact_uuids
    ]


def retract_insight(
    dag: FindingsDag, insight_id: int, cause: str, ace: AceStore | None = None
) -> PropagationReport:
    """Trigger (c): manual retraction. The insight itself is retracted; its
    dependents are flagged stale."""
    dag.set_status(insight_id, "retracted", cause=cause)
    for pid in dag.get_insight(insight_id).pattern_ids:
        dag.decrement_pattern_live(pid)
    report = propagate_invalidation(dag, "insight", str(insight_id), cause, ace)
    report.affected_patterns = sorted(
        set(report.affected_patterns) | set(dag.get_insight(insight_id).pattern_ids)
    )
    return report
