"""Search-loop termination and persistence behavior of the composition loop."""

import json

from lcm_mem.llm.gateway import CachedGateway
from lcm_mem.memory.compose import ComposeConfig, answer_query
from lcm_mem.memory.store import MemoryStore
from tests.conftest import FakeClient, default_handler


def _store_with_facts(hash_embedder, n=6):
    store = MemoryStore(":memory:", hash_embedder)
    for i in range(n):
        store.add_fact(f"fact number {i} about topic{i}", entities=[f"topic{i}"])
    return store


def test_loop_terminates_when_llm_returns_none(fake_gateway, hash_embedder):
    """All compose calls return NONE and answerability says no: the loop must
    stop on its own (budget/threshold/pair exhaustion), not hang."""
    store = _store_with_facts(hash_embedder)
    cfg = ComposeConfig(scorer="random", max_llm_calls=5, max_depth=3,
                        t_answerable=2.0, min_pair_score=-1.0)
    res = answer_query(store, "unanswerable question", fake_gateway, cfg)
    assert res.stopped_because in ("budget", "exhausted_pairs", "below_threshold")
    assert res.llm_calls <= cfg.max_llm_calls
    assert res.derived_ids == []


def test_loop_respects_budget_when_composing(tmp_path, hash_embedder):
    """LLM always produces an inference; loop must stop at max_llm_calls and
    persist the derived facts with provenance."""

    def handler(content: str) -> str:
        if content.startswith("Given fact A and fact B"):
            return json.dumps({"inference": "a fresh derived inference " + str(hash(content) % 97),
                               "confidence": 0.9, "used_world_knowledge": False})
        return default_handler(content)

    gw = CachedGateway(client=FakeClient(handler), cache_path=tmp_path / "c.sqlite")
    store = _store_with_facts(hash_embedder)
    n_before = len(store.live_fact_ids())
    cfg = ComposeConfig(scorer="random", max_llm_calls=3, max_depth=3,
                        t_answerable=2.0, min_pair_score=-1.0)
    res = answer_query(store, "some question", gw, cfg)
    assert res.llm_calls <= 3
    assert len(res.derived_ids) >= 1
    # derived facts persisted with parents and decayed confidence
    for fid in res.derived_ids:
        f = store.get_fact(fid)
        assert f.type == "derived"
        assert len(store.parents(fid)) == 2
        assert f.confidence < 0.9  # decay applied
    assert len(store.live_fact_ids()) == n_before + len(res.derived_ids)


def test_derived_facts_are_reused_across_queries(tmp_path, hash_embedder):
    def handler(content: str) -> str:
        if content.startswith("Given fact A and fact B"):
            return json.dumps({"inference": "the reusable combined insight",
                               "confidence": 0.9, "used_world_knowledge": False})
        return default_handler(content)

    gw = CachedGateway(client=FakeClient(handler), cache_path=tmp_path / "c.sqlite")
    store = _store_with_facts(hash_embedder, n=4)
    # budget 2: one answerability check + one composition
    cfg = ComposeConfig(scorer="random", max_llm_calls=2, max_depth=3,
                        t_answerable=2.0, min_pair_score=-1.0)
    res1 = answer_query(store, "the reusable combined insight", gw, cfg)
    assert res1.derived_ids
    # second query retrieves the persisted derived fact as a candidate
    res2 = answer_query(store, "the reusable combined insight", gw, cfg)
    assert set(res1.derived_ids) & set(res2.reused_derived_ids)


def test_max_depth_respected(tmp_path, hash_embedder):
    def handler(content: str) -> str:
        if content.startswith("Given fact A and fact B"):
            return json.dumps({"inference": "derived " + str(hash(content) % 1000),
                               "confidence": 1.0, "used_world_knowledge": False})
        return default_handler(content)

    gw = CachedGateway(client=FakeClient(handler), cache_path=tmp_path / "c.sqlite")
    store = _store_with_facts(hash_embedder, n=3)
    cfg = ComposeConfig(scorer="random", max_llm_calls=50, max_depth=2,
                        t_answerable=2.0, min_pair_score=-1.0)
    answer_query(store, "q", gw, cfg)
    depths = [store.get_fact(f).depth for f in store.live_fact_ids()]
    assert max(depths) <= 2


def test_world_knowledge_flag_sets_type(tmp_path, hash_embedder):
    def handler(content: str) -> str:
        if content.startswith("Given fact A and fact B"):
            return json.dumps({"inference": "bridge via world knowledge",
                               "confidence": 0.8, "used_world_knowledge": True})
        return default_handler(content)

    gw = CachedGateway(client=FakeClient(handler), cache_path=tmp_path / "c.sqlite")
    store = _store_with_facts(hash_embedder, n=3)
    cfg = ComposeConfig(scorer="random", max_llm_calls=2, max_depth=3,
                        t_answerable=2.0, min_pair_score=-1.0)
    res = answer_query(store, "q", gw, cfg)
    assert res.derived_ids
    assert store.get_fact(res.derived_ids[0]).type == "world_bridge"
