"""Generator ground-truth consistency + the offline findings suite end-to-end.

The full-suite integration test runs with the real e5 embedder (cached locally)
because analog recall is a semantic-ranking claim; pure-logic tests above it
use the hashing embedder.
"""

import pytest

from risk_agent_memory.embedding import HashingEmbedder, get_embedder
from risk_agent_memory.evals.gen import apply_restatement, build_corpus
from risk_agent_memory.evals.findings_suite import run_offline_suite
from risk_agent_memory.stores.findings.backend import InMemoryFactBackend
from risk_agent_memory.stores.findings.dag import FindingsDag
from risk_agent_memory.stores.findings.writer import AbstractionValidator

DENYLISTS = "configs/denylists.yaml"


@pytest.fixture
def corpus():
    backend = InMemoryFactBackend()
    dag = FindingsDag(":memory:", HashingEmbedder(dim=32))
    gt = build_corpus(backend, dag, AbstractionValidator.load(DENYLISTS))
    return backend, dag, gt


def test_ground_truth_structure(corpus):
    backend, dag, gt = corpus
    assert gt.analog_pairs == [("INC-A", "INC-B")]
    assert gt.distractors == ["INC-D"]
    for inc in gt.incidents.values():
        assert inc.insight_id is not None
        assert len(inc.fact_uuids) == 4
        # generator abstractions pass the validator (they are entity-free)
        assert dag.get_insight(inc.insight_id).status == "valid"


def test_analog_instances_share_one_pattern(corpus):
    _, dag, gt = corpus
    a = dag.get_insight(gt.incidents["INC-A"].insight_id)
    b = dag.get_insight(gt.incidents["INC-B"].insight_id)
    assert a.pattern_ids and a.pattern_ids == b.pattern_ids   # canonicalized, no twin
    assert len(dag.patterns()) == 1
    assert dag.get_pattern(a.pattern_ids[0]).live_instances == 2


def test_restatement_flags_only_the_restated_incident(corpus):
    backend, dag, gt = corpus
    closed = apply_restatement(backend, dag, gt, "INC-A")
    assert len(closed) == 1
    assert dag.get_insight(gt.incidents["INC-A"].insight_id).status == "flagged_stale"
    assert dag.get_insight(gt.incidents["INC-B"].insight_id).status == "valid"
    assert dag.get_insight(gt.incidents["INC-D"].insight_id).status == "valid"
    # pattern live count decremented from 2 to 1
    pid = dag.get_insight(gt.incidents["INC-A"].insight_id).pattern_ids[0]
    assert dag.get_pattern(pid).live_instances == 1


@pytest.mark.integration
def test_offline_suite_with_real_embedder():
    metrics = run_offline_suite(embedder_name="intfloat/e5-base-v2",
                                denylists_path=DENYLISTS)
    rec = metrics["cross_incident_recall"]
    assert rec["analog_hit_rate@k"] == 1.0
    assert rec["distractor_rejection_rate"] == 1.0
    inv = metrics["invalidation"]
    assert inv["stale_answer_rate"] == 0
    assert inv["over_invalidation"] == 0
    assert inv["restated_flagged"] is True
    assert metrics["temporal_qa"]["as_of_correct"] == 1.0
    assert metrics["registry_size_ok"] is True
