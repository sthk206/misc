"""C.5 retrieval: union of three surfaces, merged and deduped.

(a) backend temporal-fact search (entity + validity-window scoped);
(b) embedding search over insight ABSTRACTIONS (not narratives);
(c) pattern-node hop: situation -> pattern -> instance insights (two-hop cap).

Flagged insights are always returned WITH their flag so the agent can say
"this prior conclusion was superseded because X" (C.6 rule 5).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from risk_agent_memory.config import CONFIG
from risk_agent_memory.stores.findings.backend import FactBackend, TemporalFact
from risk_agent_memory.stores.findings.dag import FindingsDag, Insight


@dataclass
class RetrievalResult:
    facts: list[TemporalFact] = field(default_factory=list)
    insights: list[Insight] = field(default_factory=list)   # deduped union of (b) and (c)
    via_pattern: dict[int, int] = field(default_factory=dict)  # insight_id -> pattern_id
    flagged: list[Insight] = field(default_factory=list)     # subset with status != valid

    def render(self) -> str:
        lines = []
        if self.facts:
            lines.append("### Facts (validity-scoped)")
            for f in self.facts:
                to = f.valid_to if f.valid_to is not None else "open"
                lines.append(f"- [{f.uuid[:8]}] {f.subject} {f.predicate} {f.object} "
                             f"(valid {f.valid_from} -> {to})")
        if self.insights:
            lines.append("### Prior insights")
            for i in self.insights:
                flag = "" if i.status == "valid" else (
                    f" [FLAG: {i.status}" + (f" — {i.stale_cause}" if i.stale_cause else "") + "]"
                )
                via = f" (via pattern {self.via_pattern[i.id]})" if i.id in self.via_pattern else ""
                lines.append(f"- [i{i.id}]{flag}{via} {i.narrative}")
        return "\n".join(lines)


def retrieve(
    dag: FindingsDag,
    backend: FactBackend,
    situation: str,
    entities: list[str] | None = None,
    as_of: float | None = None,
    k_facts: int = 20,
    k_insights: int = 10,
    pattern_hops: int | None = None,
) -> RetrievalResult:
    pattern_hops = pattern_hops or CONFIG.thresholds.pattern_hop_cap

    # (a) temporal facts
    facts = backend.search(entities=entities, as_of=as_of, k=k_facts)

    # (b) abstraction embedding search (flagged included, with flag)
    scored = dag.search_abstractions(situation, k=k_insights, include_flagged=True)
    insights: dict[int, Insight] = {i.id: i for i, _ in scored}
    score: dict[int, float] = {i.id: s for i, s in scored}

    # (c) pattern hop: situation -> nearest pattern(s) -> instance insights.
    # Pattern-mediated reachability ADDS relevance mass (PPR-style): this is
    # what beats the textually-similar-but-causally-different distractor that
    # flat embedding search falls for.
    via_pattern: dict[int, int] = {}
    if pattern_hops >= 2:
        patterns = dag.patterns()
        if patterns:
            q = dag.embedder.encode([situation])[0]
            descs = dag.embedder.encode([p.description for p in patterns])
            sims = descs @ q
            order = np.argsort(-sims)[: max(1, k_insights // 3)]
            for pi in order:
                if sims[pi] <= 0:
                    continue
                p = patterns[int(pi)]
                for iid in p.instance_insight_ids:
                    ins = insights.get(iid) or dag.get_insight(iid)
                    if ins.status == "retracted":
                        continue
                    insights[iid] = ins
                    via_pattern[iid] = p.id
                    score[iid] = score.get(iid, 0.0) + float(sims[pi])

    merged = sorted(insights.values(), key=lambda i: -score.get(i.id, 0.0))
    merged = merged[:k_insights]
    return RetrievalResult(
        facts=facts,
        insights=merged,
        via_pattern=via_pattern,
        flagged=[i for i in merged if i.status != "valid"],
    )
