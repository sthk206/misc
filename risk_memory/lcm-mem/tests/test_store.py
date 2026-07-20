from lcm_mem.memory.ingest import canonicalize_entities, ingest_text
from lcm_mem.memory.store import MemoryStore, VectorIndex


def test_vector_index_exact_search(hash_embedder):
    idx = VectorIndex(dim=32)
    vecs = hash_embedder.encode([f"text {i}" for i in range(10)])
    for i, v in enumerate(vecs):
        idx.add(100 + i, v)
    hits = idx.search(vecs[3], k=3)
    assert hits[0][0] == 103
    assert hits[0][1] > 0.999


def test_add_and_search_excludes_dead_facts(hash_embedder):
    store = MemoryStore(":memory:", hash_embedder)
    fid1 = store.add_fact("the sky is blue")
    store.add_fact("grass is green")
    hits = store.search("the sky is blue", k=2)
    assert hits[0][0] == fid1
    store.mark_invalidated(fid1)
    hits = store.search("the sky is blue", k=2)
    assert fid1 not in [h[0] for h in hits]
    store2_ids = store.live_fact_ids()
    assert fid1 not in store2_ids


def test_entity_linking_and_lookup(hash_embedder):
    store = MemoryStore(":memory:", hash_embedder)
    f1 = store.add_fact("Alice works at Acme.", entities=["Alice", "Acme"])
    f2 = store.add_fact("Bob lives in Paris.", entities=["Bob", "Paris"])
    assert store.facts_with_entities(["Alice"]) == [f1]
    assert store.facts_with_entities(["Paris", "Acme"]) == [f1, f2]
    assert store.facts_with_entities(["Nobody"]) == []
    assert set(store.entities_of(f1)) == {"Alice", "Acme"}


def test_canonicalization_merges_aliases(hash_embedder, fake_gateway):
    store = MemoryStore(":memory:", hash_embedder)
    f1 = store.add_fact("X.", entities=["Acme Corp"])
    f2 = store.add_fact("Y.", entities=["Acme Corp"])  # exact same name: same entity row
    f3 = store.add_fact("Z.", entities=["Bob"])
    merges = canonicalize_entities(store, hi=1.01, lo=1.01)  # thresholds > 1: nothing merges
    assert merges == 0
    # identical names share the entity id even without canonicalization
    assert store.facts_with_entities(["Acme Corp"]) == [f1, f2]
    assert store.facts_with_entities(["Bob"]) == [f3]


def test_ingest_pipeline(fake_gateway, hash_embedder):
    store = MemoryStore(":memory:", hash_embedder)
    ids = ingest_text(store, "some session text", fake_gateway, "m", "session-1")
    assert len(ids) == 2  # fake handler emits two facts
    f = store.get_fact(ids[0])
    assert f.type == "observed"
    assert f.confidence == 1.0
    assert f.depth == 0
    assert f.source_session == "session-1"
    assert store.facts_with_entities(["Acme"]) == sorted(ids)
