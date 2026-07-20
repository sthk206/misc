from lcm_mem.data.musique import (
    assign_groups,
    component_ids,
    extract_triples_for_example,
    split_by_group,
    substitute_placeholders,
)


def _example(eid: str, hops: int, answerable: bool = True) -> dict:
    decomp = []
    for i in range(hops):
        q = f"sub-question {i}" if i == 0 else f"sub-question {i} about #{i}"
        decomp.append({"question": q, "answer": f"answer{i}",
                       "paragraph_support_idx": i})
    return {
        "id": eid,
        "question": "the full question?",
        "answer": "final answer",
        "answerable": answerable,
        "question_decomposition": decomp,
    }


def test_component_ids():
    assert component_ids("2hop__123_456") == ["123", "456"]
    assert component_ids("4hop3__1_2_3_4") == ["1", "2", "3", "4"]


def test_placeholder_substitution():
    assert substitute_placeholders("Who leads #1?", ["France"]) == "Who leads France?"
    assert substitute_placeholders("no refs", ["x"]) == "no refs"


def test_group_assignment_links_shared_components():
    groups = assign_groups(["2hop__1_2", "2hop__2_3", "2hop__4_5"])
    assert groups["2hop__1_2"] == groups["2hop__2_3"]
    assert groups["2hop__1_2"] != groups["2hop__4_5"]


def test_split_no_single_hop_leakage():
    """Examples sharing a single-hop component must land in the same split."""
    # pairs of examples share one component (i+1); pairs are disjoint from
    # each other, giving ~100 independent leakage groups
    ids = [f"2hop__{i}_{i + 1}" for i in range(0, 400, 4)]
    ids += [f"2hop__{i + 1}_{i + 2}" for i in range(0, 400, 4)]
    split = split_by_group(ids, val_frac=0.2, test_frac=0.2)
    comp_split: dict[str, str] = {}
    for eid in ids:
        for c in component_ids(eid):
            assert comp_split.setdefault(c, split[eid]) == split[eid], (
                f"component {c} appears in {comp_split[c]} and {split[eid]}"
            )
    assert len(set(split.values())) > 1  # sanity: split actually splits


def test_two_hop_triple_extraction(fake_gateway):
    triples = extract_triples_for_example(
        _example("2hop__10_20", 2), fake_gateway, "m", group_id="g"
    )
    assert len(triples) == 1
    t = triples[0]
    assert t.fact_a == "sub-question 0 is answer0."
    # placeholder #1 resolved with hop-1's answer before rewriting
    assert t.fact_b == "sub-question 1 about answer0 is answer1."
    assert t.composed_gt == "the full question is final answer."
    assert t.query == "the full question?"
    assert t.answerable


def test_chained_extraction_for_three_hop(fake_gateway):
    triples = extract_triples_for_example(
        _example("3hop1__1_2_3", 3), fake_gateway, "m", group_id="g"
    )
    assert len(triples) == 2
    # pair 0 composes hops 1+2 into a bridge; pair 1 uses the bridge as fact_a
    assert triples[0].pair_depth == 0
    assert triples[1].pair_depth == 1
    assert triples[1].fact_a == triples[0].composed_gt
    # the final pair's target is the declarative form of the full question
    assert triples[1].composed_gt == "the full question is final answer."


def test_unanswerable_flag_propagates(fake_gateway):
    triples = extract_triples_for_example(
        _example("2hop__7_8", 2, answerable=False), fake_gateway, "m", group_id="g"
    )
    assert triples and not triples[0].answerable


def test_rewrite_calls_are_cached(fake_gateway):
    ex = _example("2hop__10_20", 2)
    extract_triples_for_example(ex, fake_gateway, "m", group_id="g")
    calls_first = fake_gateway._test_client.calls
    extract_triples_for_example(ex, fake_gateway, "m", group_id="g")
    assert fake_gateway._test_client.calls == calls_first
