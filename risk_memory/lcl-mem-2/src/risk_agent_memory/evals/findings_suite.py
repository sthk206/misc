"""6.4 Phase C suite: "recall the right precedent, respect time, retract
cleanly." The retrieval/invalidation/temporal metrics run fully offline
against generator ground truth; amortization and LLM-judged insight quality
need live sessions (harness)."""

from __future__ import annotations

from risk_agent_memory.evals.gen import GroundTruth, apply_restatement, build_corpus
from risk_agent_memory.stores.findings.backend import InMemoryFactBackend
from risk_agent_memory.stores.findings.dag import FindingsDag
from risk_agent_memory.stores.findings.retrieval import retrieve
from risk_agent_memory.stores.findings.writer import AbstractionValidator


def situation_query(gt: GroundTruth, incident_id: str) -> str:
    """'Have we seen this before?' query built from the incident narrative —
    what the agent would form when the new incident surfaces."""
    inc = gt.incidents[incident_id]
    return (
        f"Investigating {inc.pair} on {inc.desk}: a hedge appears to remain "
        f"open after its option expired. Have we seen this pattern before?"
    )


def cross_incident_recall(
    dag: FindingsDag, backend: InMemoryFactBackend, gt: GroundTruth, k: int = 5
) -> dict:
    """THE metric the DAG exists for: querying from the later incident of each
    analog pair must surface the earlier one; distractors must rank below."""
    hits = 0
    distractor_intrusions = 0
    n = 0
    for early_id, late_id in gt.analog_pairs:
        n += 1
        res = retrieve(
            dag, backend, situation_query(gt, late_id),
            entities=[gt.incidents[late_id].pair], k_insights=k,
        )
        got_ids = [i.id for i in res.insights][:k]
        if gt.incidents[early_id].insight_id in got_ids:
            hits += 1
        distractor_insights = {
            gt.incidents[d].insight_id for d in gt.distractors
        }
        early_rank = (
            got_ids.index(gt.incidents[early_id].insight_id)
            if gt.incidents[early_id].insight_id in got_ids else len(got_ids)
        )
        for iid in got_ids[:early_rank]:
            if iid in distractor_insights:
                distractor_intrusions += 1
    return {
        "analog_hit_rate@k": hits / max(n, 1),
        "distractor_rejection_rate": 1 - distractor_intrusions / max(n, 1),
        "k": k,
        "n_pairs": n,
    }


def invalidation_correctness(
    dag: FindingsDag, backend: InMemoryFactBackend, gt: GroundTruth, ace=None
) -> dict:
    """Inject a restatement, re-run retrieval, and check:
    - stale-answer rate: restated conclusions returned WITHOUT a flag (fail);
    - over-invalidation: untouched insights wrongly flagged."""
    restated = "INC-A"
    apply_restatement(backend, dag, gt, restated, ace=ace)

    res = retrieve(dag, backend, situation_query(gt, restated),
                   entities=[gt.incidents[restated].pair])
    stale_unflagged = 0
    restated_iid = gt.incidents[restated].insight_id
    for i in res.insights:
        if i.id == restated_iid and i.status == "valid":
            stale_unflagged += 1

    over_invalidated = sum(
        1
        for iid, inc in ((v.insight_id, v) for v in gt.incidents.values())
        if inc.incident_id not in gt.restated_incidents
        and dag.get_insight(iid).status != "valid"
    )
    pattern_ids = dag.get_insight(restated_iid).pattern_ids
    return {
        "stale_answer_rate": stale_unflagged,          # target 0
        "over_invalidation": over_invalidated,          # target 0
        "restated_flagged": dag.get_insight(restated_iid).status == "flagged_stale",
        "pattern_live_after": [dag.get_pattern(p).live_instances for p in pattern_ids],
    }


def temporal_qa(backend: InMemoryFactBackend, gt: GroundTruth) -> dict:
    """As-of retrieval against generator ground truth: what was believed about
    the option's status before vs after expiry vs after restatement."""
    inc = gt.incidents["INC-B"]
    checks = []
    pre = backend.search(entities=[inc.option_trade], as_of=inc.day - 1)
    checks.append(all(
        f.object != "expired" for f in pre if f.predicate == "status"
    ))
    post = backend.search(entities=[inc.option_trade], as_of=inc.day + 0.5)
    checks.append(any(
        f.predicate == "status" and f.object == "expired" for f in post
    ))
    return {"as_of_correct": sum(checks) / len(checks), "n_checks": len(checks)}


def run_offline_suite(embedder_name: str = "intfloat/e5-base-v2",
                      denylists_path: str = "configs/denylists.yaml") -> dict:
    from risk_agent_memory.embedding import get_embedder

    embedder = get_embedder(embedder_name)
    backend = InMemoryFactBackend()
    dag = FindingsDag(":memory:", embedder)
    validator = AbstractionValidator.load(denylists_path)
    gt = build_corpus(backend, dag, validator)

    metrics = {
        "cross_incident_recall": cross_incident_recall(dag, backend, gt),
        "temporal_qa": temporal_qa(backend, gt),
        # run invalidation LAST — it mutates the corpus
        "invalidation": invalidation_correctness(dag, backend, gt),
        "registry_size_ok": dag.registry_size_ok(),
    }
    return metrics
