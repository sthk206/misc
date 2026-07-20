"""C.7 promotion — the ONLY path from findings to playbook (single write path
rule: the reflector must not independently distill incident conclusions).

Nightly job: patterns crossing the bar (>= 2 valid instances, OR 1 instance
with severity above threshold) -> draft an ACE candidate entry with
justification_ptr=pattern:<id> -> Phase A approval flow.
"""

from __future__ import annotations

from risk_agent_memory.config import CONFIG
from risk_agent_memory.stores.ace.models import AceStore
from risk_agent_memory.stores.findings.dag import FindingsDag


def draft_directive(pattern_name: str, description: str) -> str:
    """Deterministic draft; the human edits at approval time."""
    text = f"Watch for {pattern_name}: {description}"
    return text if len(text) <= 220 else text[:217] + "..."


def scan_and_promote(dag: FindingsDag, ace: AceStore) -> list[int]:
    """Returns delta-queue ids of drafted ACE candidates."""
    t = CONFIG.thresholds
    drafted: list[int] = []
    already = {
        e.justification_ptr
        for e in ace.entries()
        if e.justification_ptr and e.status != "retired"
    }
    pending_ptrs = {
        (d["payload"] or {}).get("justification_ptr")
        for d in ace.pending_deltas()
    }
    for p in dag.patterns():
        ptr = f"pattern:{p.id}"
        if ptr in already or ptr in pending_ptrs or p.status != "active":
            continue
        max_sev = 0.0
        for iid in p.instance_insight_ids:
            ins = dag.get_insight(iid)
            if ins.status == "valid":
                max_sev = max(max_sev, ins.severity)
        crosses = p.live_instances >= t.promotion_min_instances or (
            p.live_instances >= 1 and max_sev >= t.promotion_severity_override
        )
        if not crosses:
            continue
        delta_id = ace.submit_delta(
            "ADD",
            {
                "text": draft_directive(p.name, p.description),
                "scope": "global",
                "justification_ptr": ptr,
                "evidence_span": f"pattern {p.id} with {p.live_instances} live instances",
            },
            session_ref="promotion:nightly",
        )
        drafted.append(delta_id)
    return drafted
