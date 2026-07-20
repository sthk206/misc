import pytest

from risk_agent_memory.stores.ace.compiler import compile_playbook
from risk_agent_memory.stores.ace.models import EntryTooLong, approx_tokens
from risk_agent_memory.stores.ace.reflector import apply_reflection, parse_deltas


def _seed(ace, text, scope="global", **kw):
    return ace.add_entry(text, scope=scope, status="active",
                         created_by="human", approved_by="test", **kw)


def test_entry_token_cap(ace):
    with pytest.raises(EntryTooLong):
        ace.add_entry("word " * 200)


def test_active_requires_approval(ace):
    with pytest.raises(ValueError):
        ace.add_entry("Check hedges.", status="active")
    eid = ace.add_entry("Check hedges.", status="candidate")
    with pytest.raises(ValueError):
        ace.set_status(eid, "active")
    ace.set_status(eid, "active", by="human")
    assert ace.get(eid).status == "active"


def test_compile_scope_matching_and_isolation(ace):
    a = _seed(ace, "Rule for manager A only.", scope="manager:mgr_a")
    g = _seed(ace, "Global rule.", scope="global")
    m = _seed(ace, "Morning rule.", scope="mode:morning")
    pb_a = compile_playbook(ace, "mgr_a", "morning")
    ids_a = [e.id for e in pb_a.included]
    assert set(ids_a) == {a, g, m}
    # specificity ordering: manager > mode > global
    assert ids_a.index(a) < ids_a.index(m) < ids_a.index(g)
    # manager B never sees A's rule
    pb_b = compile_playbook(ace, "mgr_b", "adhoc")
    assert a not in [e.id for e in pb_b.included]
    assert m not in [e.id for e in pb_b.included]


def test_budget_enforcement_drops_lowest_scoring(ace):
    keep = _seed(ace, "High value rule that fires all the time.")
    ace.incr(keep, "helpful")
    ace.incr(keep, "helpful")
    junk_ids = [
        _seed(ace, f"Junk directive number {i} " + "padding words here " * 8)
        for i in range(60)
    ]
    for j in junk_ids[:10]:
        ace.incr(j, "harmful")
    pb = compile_playbook(ace, "mgr_a", "morning", budget=300)
    assert pb.tokens <= 300 + approx_tokens("## Playbook\n")
    assert keep in [e.id for e in pb.included]
    assert pb.dropped
    # dropped entries were flagged to the pruning queue
    flagged = {
        r["entry_id"] for r in ace.db.execute("SELECT entry_id FROM pruning_queue")
    }
    assert {e.id for e in pb.dropped} <= flagged
    # harmful-scored junk dropped before neutral junk
    dropped_ids = {e.id for e in pb.dropped}
    assert set(junk_ids[:10]) <= dropped_ids


def test_dedup_gate_converts_add_to_incr(ace):
    eid = _seed(ace, "Always check for orphaned hedges after expiries.")
    before = ace.get(eid).helpful_count
    ace.submit_delta(
        "ADD",
        {"text": "Always check for orphaned hedges after expiries.",
         "scope": "global", "evidence_span": "missed one today"},
    )
    assert ace.get(eid).helpful_count == before + 1
    assert ace.pending_deltas() == []          # nothing queued
    # a genuinely different directive DOES queue
    ace.submit_delta("ADD", {"text": "Completely unrelated new directive.",
                             "scope": "global"})
    assert len(ace.pending_deltas()) == 1


def test_approval_flow(ace):
    did = ace.submit_delta("ADD", {"text": "New rule from reflection.",
                                   "scope": "mode:morning"})
    assert ace.entries("active") == []
    eid = ace.decide_delta(did, approve=True, by="reviewer")
    e = ace.get(eid)
    assert e.status == "active" and e.approved_by == "reviewer"
    # rejected deltas change nothing
    did2 = ace.submit_delta("ADD", {"text": "Bad idea rule.", "scope": "global"})
    assert ace.decide_delta(did2, approve=False, by="reviewer") is None
    assert len(ace.entries("active")) == 1


def test_merge_carries_counters_and_retires_sources(ace):
    e1 = _seed(ace, "Check option expiries in the morning.")
    e2 = _seed(ace, "Review expiring options early.")
    ace.incr(e1, "helpful")
    ace.incr(e2, "helpful")
    did = ace.submit_delta("MERGE", {"entry_ids": [e1, e2],
                                     "text": "Review option expiries every morning."})
    new_id = ace.decide_delta(did, approve=True, by="reviewer")
    merged = ace.get(new_id)
    assert merged.helpful_count == 2
    assert ace.get(e1).status == "retired"
    assert ace.get(e2).status == "retired"


def test_reflector_parse_and_routing(ace, prefs):
    raw = """```json
    [{"kind": "ADD", "text": "Check restatements daily.", "scope": "global",
      "evidence_span": "missed a cancel"},
     {"kind": "INCR", "entry_id": %d, "direction": "helpful",
      "evidence_span": "caught the orphan"},
     {"kind": "PREF_CANDIDATE", "manager_id": "mgr_a",
      "key": "layout.chart_order", "value": ["EURUSD"],
      "evidence_span": "asked three times"}]
    ```"""
    eid = _seed(ace, "Always check for orphaned hedges.")
    deltas = parse_deltas(raw % eid)
    counts = apply_reflection(ace, deltas, "sess-1", prefs_store=prefs)
    assert counts == {"ADD": 1, "INCR": 1, "PREF_CANDIDATE": 1}
    assert ace.get(eid).helpful_count == 1
    assert len(ace.pending_deltas()) == 1
    cands = prefs.candidates("mgr_a")
    assert len(cands) == 1 and cands[0].key == "layout.chart_order"
    # candidate is NOT in the confirmed profile
    assert prefs.profile("mgr_a") == {}
