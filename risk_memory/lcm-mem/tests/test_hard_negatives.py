"""Each hard negative must differ from the positive in exactly the intended way."""

import random
import re

from lcm_mem.data.hard_negatives import (
    build_entity_pool,
    generate_hard_negatives,
    negate,
    role_swap,
    swap_date,
    swap_entity,
    swap_number,
)

RNG = random.Random(0)


def _token_diff(a: str, b: str) -> tuple[list[str], list[str]]:
    ta, tb = a.split(), b.split()
    return ([t for t in ta if t not in tb], [t for t in tb if t not in ta])


def test_date_swap_changes_only_the_year():
    text = "The treaty was signed in 1848 in Paris."
    out = swap_date(text, RNG)
    assert out is not None and out != text
    removed, added = _token_diff(text, out)
    assert len(removed) == 1 and len(added) == 1
    assert removed[0].isdigit() and added[0].isdigit()


def test_date_swap_month():
    out = swap_date("The launch happened in March.", RNG)
    assert out is not None
    assert "March" not in out
    assert any(m in out for m in ("January", "February", "April", "May", "June",
                                  "July", "August", "September", "October",
                                  "November", "December"))


def test_number_swap_skips_years_changes_number():
    text = "The company hired 120 engineers in 2019."
    out = swap_number(text, RNG)
    assert out is not None
    assert "2019" in out          # year untouched
    assert "120" not in out       # count perturbed
    removed, added = _token_diff(text, out)
    assert len(removed) == 1 and len(added) == 1


def test_negation_insertion_and_removal_roundtrip():
    pos = "The bridge is open to traffic."
    neg = negate(pos)
    assert neg == "The bridge is not open to traffic."
    assert negate(neg) == pos     # removal path


def test_negation_none_when_not_applicable():
    assert negate("Sunrise over mountains.") is None


def test_role_swap():
    assert role_swap("Alice hired Bob.") == "Bob hired Alice."
    assert role_swap("Alice hired Bob") == "Bob hired Alice"
    assert role_swap("The sky is blue.") is None


def test_entity_swap_uses_same_type_pool():
    pool = {"SPAN": ["Alice Smith", "Carol Jones", "Dan Brown"]}
    text = "Alice Smith won the award."
    out = swap_entity(text, pool, random.Random(1))
    assert out is not None and out != text
    assert "Alice Smith" not in out
    assert any(e in out for e in ("Carol Jones", "Dan Brown"))


def test_build_entity_pool_and_generate():
    texts = ["Alice Smith visited Berlin in 1990.", "Carol Jones lives in Madrid."]
    pool = build_entity_pool(texts)
    negs = generate_hard_negatives(texts[0], pool, random.Random(2))
    kinds = {n.kind for n in negs}
    assert "date_swap" in kinds
    assert all(n.text != texts[0] for n in negs)
    # every generated negative differs in exactly one intended way, so at most
    # one of each kind
    assert len(kinds) == len(negs)


def test_generated_negatives_preserve_everything_else():
    text = "The merger was completed in 2005."
    for neg in generate_hard_negatives(text, None, random.Random(3)):
        if neg.kind == "date_swap":
            assert re.sub(r"\d{4}", "Y", neg.text) == re.sub(r"\d{4}", "Y", text)
        if neg.kind == "negation":
            assert neg.text.replace(" not", "") == text
