"""Property-based invalidation tests: invalidating a fact marks stale all and
only its provenance descendants."""

from hypothesis import given, settings
from hypothesis import strategies as st

from lcm_mem.encoder.embed import HashingEmbedder
from lcm_mem.memory.provenance import (
    check_and_propagate,
    descendant_closure,
    invalidate_fact,
)
from lcm_mem.memory.store import MemoryStore


@st.composite
def random_dag(draw):
    """Random DAG as {child: [parents]} over nodes 0..n-1 where parents < child
    (guarantees acyclicity)."""
    n = draw(st.integers(min_value=2, max_value=25))
    parents_of = {}
    for child in range(1, n):
        k = draw(st.integers(min_value=0, max_value=min(child, 3)))
        parents_of[child] = sorted(
            draw(
                st.sets(st.integers(min_value=0, max_value=child - 1),
                        min_size=k, max_size=k)
            )
        )
    return n, parents_of


def _children_of(parents_of: dict[int, list[int]]) -> dict[int, list[int]]:
    ch: dict[int, list[int]] = {}
    for c, ps in parents_of.items():
        for p in ps:
            ch.setdefault(p, []).append(c)
    return ch


@given(random_dag(), st.data())
@settings(max_examples=60, deadline=None)
def test_closure_is_all_and_only_descendants(dag, data):
    n, parents_of = dag
    children = _children_of(parents_of)
    root = data.draw(st.integers(min_value=0, max_value=n - 1))
    closure = descendant_closure(children, [root])

    # brute-force reference: node is a descendant iff a parent-path reaches root
    def is_descendant(x: int) -> bool:
        stack, seen = [x], set()
        while stack:
            y = stack.pop()
            for p in parents_of.get(y, []):
                if p == root:
                    return True
                if p not in seen:
                    seen.add(p)
                    stack.append(p)
        return False

    expected = {x for x in range(n) if x != root and is_descendant(x)}
    assert closure == expected


@given(random_dag(), st.data())
@settings(max_examples=25, deadline=None)
def test_store_invalidation_marks_exactly_descendants(dag, data):
    n, parents_of = dag
    store = MemoryStore(":memory:", HashingEmbedder(dim=16))
    ids = {}
    for node in range(n):
        parents = [ids[p] for p in parents_of.get(node, [])]
        ids[node] = store.add_fact(
            f"fact {node}",
            type="observed" if not parents else "derived",
            parents=parents,
        )
    root = data.draw(st.integers(min_value=0, max_value=n - 1))
    stale = invalidate_fact(store, ids[root])

    expected = descendant_closure(_children_of(parents_of), [root])
    assert stale == {ids[x] for x in expected}
    # invalidated root is hard-dead, descendants soft-stale, others live
    assert not store.is_live(ids[root])
    for x in range(n):
        if x == root:
            continue
        assert store.is_live(ids[x]) == (x not in expected)
        if x in expected:
            f = store.get_fact(ids[x])
            assert f.stale and f.invalidated_at is None  # soft, reversible


def test_depth_computed_from_parents():
    store = MemoryStore(":memory:", HashingEmbedder(dim=16))
    a = store.add_fact("a")
    b = store.add_fact("b")
    d1 = store.add_fact("d1", type="derived", parents=[a, b])
    d2 = store.add_fact("d2", type="derived", parents=[d1, a])
    assert store.get_fact(a).depth == 0
    assert store.get_fact(d1).depth == 1
    assert store.get_fact(d2).depth == 2


def test_check_and_propagate_unrelated_touches_nothing(fake_gateway, hash_embedder):
    store = MemoryStore(":memory:", hash_embedder)
    f1 = store.add_fact("Alice lives in Paris.", entities=["Alice", "Paris"])
    f2 = store.add_fact("Alice lives in Paris today.", entities=["Alice", "Paris"])
    # fake gateway classifies every pair as unrelated -> nothing invalidated
    out = check_and_propagate(store, f2, fake_gateway, "m", sim_threshold=-1.0)
    assert out["invalidated"] == []
    assert store.is_live(f1) and store.is_live(f2)


def test_check_and_propagate_updates_invalidate(hash_embedder, tmp_path):
    import json

    from lcm_mem.llm.gateway import CachedGateway
    from tests.conftest import FakeClient

    def handler(content: str) -> str:
        n = content.count("OLD:")
        return json.dumps(["updates"] * n)

    gw = CachedGateway(client=FakeClient(handler), cache_path=tmp_path / "c.sqlite")
    store = MemoryStore(":memory:", hash_embedder)
    old = store.add_fact("Bob works at Acme.", entities=["Bob", "Acme"])
    derived = store.add_fact("Bob commutes to the Acme office.",
                             type="derived", parents=[old], entities=["Bob"])
    new = store.add_fact("Bob works at Initech.", entities=["Bob"])
    out = check_and_propagate(store, new, gw, "m", sim_threshold=-1.0)
    assert old in out["invalidated"]
    assert derived in out["stale"]
    assert not store.is_live(old)
    assert not store.is_live(derived)
    assert store.is_live(new)
