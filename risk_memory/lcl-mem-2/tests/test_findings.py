import pytest

from risk_agent_memory.stores.findings.dag import NoParentsError
from risk_agent_memory.stores.findings.invalidation import (
    propagate_invalidation,
    retract_insight,
)
from risk_agent_memory.stores.findings.promotion import scan_and_promote
from risk_agent_memory.stores.findings.retrieval import retrieve
from risk_agent_memory.stores.findings.writer import InsightDraft, write_insight


def _parents(*refs, ptype="zep_fact"):
    return [{"type": ptype, "ref": str(r), "as_of": 1.0} for r in refs]


def _insight(dag, abstraction="General causal statement.", parents=None, **kw):
    return dag.add_insight(
        narrative="Narrative.", abstraction=abstraction,
        claims=[{"text": "c", "epistemic": "observed", "conf": 1.0}],
        parents=parents or _parents("f1"), **kw,
    )


# ------------------------------------------------------------ backend

def test_backend_temporal_closing(backend):
    u1, closed = backend.add_fact("T1", "status", "open", valid_from=1.0)
    assert closed == []
    u2, closed = backend.add_fact("T1", "status", "expired", valid_from=5.0)
    assert closed == [u1]
    assert backend.get(u1).valid_to == 5.0
    # as-of queries respect windows
    assert [f.uuid for f in backend.search(entities=["T1"], as_of=3.0)] == [u1]
    assert [f.uuid for f in backend.search(entities=["T1"], as_of=6.0)] == [u2]


# ------------------------------------------------------------ dag / writer

def test_no_insight_without_parents(dag):
    with pytest.raises(NoParentsError):
        dag.add_insight(narrative="n", abstraction="a", claims=[], parents=[])


def test_depth_follows_insight_parents(dag):
    i1 = _insight(dag)
    i2 = _insight(dag, parents=_parents(i1, ptype="insight"))
    i3 = _insight(dag, parents=_parents(i2, ptype="insight") + _parents("f9"))
    assert dag.get_insight(i1).depth == 1
    assert dag.get_insight(i2).depth == 2
    assert dag.get_insight(i3).depth == 3


@pytest.mark.parametrize("bad", [
    "EURUSD spot dropped after the expiry.",
    "The FX Options Desk left a hedge open.",
    "Client CL-1001 concentration exceeded limits.",
    "Hedges lapsed after May 3, 2024.",
    "The move happened on 2024-05-03.",
    "This repeated the 2019 episode.",
])
def test_abstraction_validator_rejects_entities_and_dates(validator, bad):
    assert not validator.validate(bad).ok


def test_abstraction_validator_accepts_general_statement(validator):
    ok = validator.validate(
        "A hedge outliving its underlying position converts a hedged book "
        "into an outright directional exposure."
    )
    assert ok.ok, ok.reasons


def test_write_insight_rewrite_then_needs_review(dag, validator):
    draft = InsightDraft(
        narrative="n", abstraction="EURUSD hedge left open after expiry.",
        claims=[], parents=_parents("f1"),
    )
    # rewrite fixes it -> valid
    iid, status = write_insight(
        dag, draft, validator,
        rewrite=lambda text, reasons: "A hedge was left open after expiry.",
    )
    assert status == "valid"
    # no rewrite available -> stored as needs_review, never silently valid
    draft2 = InsightDraft(
        narrative="n", abstraction="USDJPY hedge left open after expiry.",
        claims=[], parents=_parents("f2"),
    )
    iid2, status2 = write_insight(dag, draft2, validator)
    assert status2 == "needs_review"
    assert dag.get_insight(iid2).status == "needs_review"


def test_generality_check_gate(dag, validator):
    draft = InsightDraft(
        narrative="n", abstraction="A perfectly entity-free statement.",
        claims=[], parents=_parents("f1"),
    )
    _, status = write_insight(dag, draft, validator, generality_check=lambda a: False)
    assert status == "needs_review"


# ------------------------------------------------------------ patterns

def test_pattern_canonicalization_links_identical(dag):
    i1 = _insight(dag)
    pid_or_none, outcome = dag.canonicalize_pattern("p", "hedge outlives option", i1)
    assert outcome == "review"                       # first proposal -> review
    pid = dag.approve_pattern(dag.pattern_review_queue()[0]["id"])
    i2 = _insight(dag)
    linked, outcome2 = dag.canonicalize_pattern("p2", "hedge outlives option", i2)
    assert outcome2 == "linked" and linked == pid
    assert dag.get_pattern(pid).live_instances == 2
    # dissimilar description -> new review item, not a link
    _, outcome3 = dag.canonicalize_pattern("q", "completely different phenomenon", i2)
    assert outcome3 == "review"


# ------------------------------------------------------------ invalidation

def _chain(dag):
    """fact f1 -> i1 -> i2 -> i3, plus unrelated i4."""
    i1 = _insight(dag, parents=_parents("f1"))
    i2 = _insight(dag, parents=_parents(i1, ptype="insight"))
    i3 = _insight(dag, parents=_parents(i2, ptype="insight"))
    i4 = _insight(dag, parents=_parents("f_other"))
    return i1, i2, i3, i4


def test_invalidation_flags_exactly_the_dependents(dag):
    i1, i2, i3, i4 = _chain(dag)
    report = propagate_invalidation(dag, "zep_fact", "f1", cause="restated")
    assert set(report.flagged_insights) == {i1, i2, i3}
    for iid in (i1, i2, i3):
        ins = dag.get_insight(iid)
        assert ins.status == "flagged_stale"
        assert ins.stale_cause == "restated"
    assert dag.get_insight(i4).status == "valid"     # no over-invalidation


def test_invalidation_weakens_ace_justification(dag, ace):
    i1 = _insight(dag, parents=_parents("f1"))
    dag.canonicalize_pattern("p", "desc", i1)
    pid = dag.approve_pattern(dag.pattern_review_queue()[0]["id"])
    eid = ace.add_entry("Watch for the pattern.", status="active",
                        created_by="promotion", approved_by="h",
                        justification_ptr=f"pattern:{pid}")
    report = propagate_invalidation(dag, "zep_fact", "f1", cause="restated", ace=ace)
    assert pid in report.affected_patterns
    assert dag.get_pattern(pid).live_instances == 0
    assert ace.get(eid).status == "candidate"        # demoted, needs reconfirm
    notes = ace.open_notifications()
    assert notes and notes[0]["kind"] == "evidence_weakened"


def test_retraction_flags_dependents(dag):
    i1, i2, i3, i4 = _chain(dag)
    report = retract_insight(dag, i1, cause="manual retraction")
    assert dag.get_insight(i1).status == "retracted"
    assert set(report.flagged_insights) == {i2, i3}


# ------------------------------------------------------------ retrieval

def test_retrieval_never_hides_flagged_insights(dag, backend):
    i1 = _insight(dag, abstraction="the exact situation text")
    propagate_invalidation(dag, "zep_fact", "f1", cause="superseded because X")
    res = retrieve(dag, backend, "the exact situation text")
    assert any(i.id == i1 for i in res.insights)
    assert any(i.id == i1 for i in res.flagged)
    assert "FLAG" in res.render() and "superseded because X" in res.render()


def test_retrieval_pattern_hop(dag, backend):
    i1 = _insight(dag, abstraction="alpha alpha alpha")
    dag.canonicalize_pattern("p", "the pattern description text", i1)
    dag.approve_pattern(dag.pattern_review_queue()[0]["id"])
    # query matches the PATTERN description, not the insight abstraction:
    # the instance is reachable only via the pattern hop
    res = retrieve(dag, backend, "the pattern description text", k_insights=1)
    assert any(i.id == i1 for i in res.insights)
    assert i1 in res.via_pattern


# ------------------------------------------------------------ promotion

def test_promotion_crosses_bar_and_dedupes(dag, ace):
    i1 = _insight(dag, parents=_parents("f1"))
    i2 = _insight(dag, parents=_parents("f2"))
    dag.canonicalize_pattern("orphaned-hedge", "hedge outlives option", i1)
    pid = dag.approve_pattern(dag.pattern_review_queue()[0]["id"])
    drafted = scan_and_promote(dag, ace)
    assert drafted == []                              # 1 instance: below bar
    dag.canonicalize_pattern("x", "hedge outlives option", i2)
    drafted = scan_and_promote(dag, ace)
    assert len(drafted) == 1                          # 2 instances: crosses
    [d] = ace.pending_deltas()
    assert d["payload"]["justification_ptr"] == f"pattern:{pid}"
    assert scan_and_promote(dag, ace) == []           # no duplicate drafts


def test_promotion_severity_override(dag, ace):
    i1 = _insight(dag, parents=_parents("f1"), severity=9.5)
    dag.canonicalize_pattern("severe", "one-off severe pattern", i1)
    dag.approve_pattern(dag.pattern_review_queue()[0]["id"])
    assert len(scan_and_promote(dag, ace)) == 1       # 1 instance but severe
