"""MuSiQue download + parse + composition-triple extraction.

For each answerable 2-hop example we produce:
  fact_a      declarative form of hop-1 (sub-question + answer as a statement)
  fact_b      declarative form of hop-2 (with #N placeholders substituted)
  composed_gt declarative form of the full question + answer
  query       the original question text
3-4 hop items emit chained pairs: (f1,f2)->bridge12, (bridge12,f3)->bridge123, ...
The final pair's target is the declarative form of the full question; intermediate
bridges are produced by a cached LLM composition of the pair.

Splits are made on the underlying single-hop question ids (union-find over shared
components) so no single-hop component leaks across train/val/test.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

from lcm_mem.llm import prompts
from lcm_mem.llm.gateway import CachedGateway

HF_DATASET = "dgslibisey/MuSiQue"

_PLACEHOLDER = re.compile(r"#(\d+)")


@dataclass
class Triple:
    qid: str            # source MuSiQue id (+ pair index for chained pairs)
    group_id: str       # leakage group (union-find over single-hop components)
    hops: int
    fact_a: str
    fact_b: str
    composed_gt: str
    query: str
    answerable: bool    # False = unanswerable variant, kept as pair-level hard negative
    pair_depth: int = 0  # 0 = first pair; k = bridge of depth k


def component_ids(example_id: str) -> list[str]:
    """'2hop__123_456' -> ['123', '456']. Components are the single-hop ids."""
    _, _, tail = example_id.partition("__")
    return [c for c in tail.split("_") if c]


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def assign_groups(example_ids: Iterable[str]) -> dict[str, str]:
    """Map every example id to a leakage group: examples sharing any single-hop
    component id land in the same group."""
    uf = UnionFind()
    comps: dict[str, list[str]] = {}
    for eid in example_ids:
        cs = component_ids(eid)
        comps[eid] = cs
        for c in cs[1:]:
            uf.union(cs[0], c)
    return {eid: uf.find(cs[0]) if cs else eid for eid, cs in comps.items()}


def split_by_group(
    example_ids: list[str],
    val_frac: float = 0.05,
    test_frac: float = 0.05,
) -> dict[str, str]:
    """Deterministic group-level split: every example in a group gets the same
    split, chosen by hashing the group id."""
    groups = assign_groups(example_ids)
    out: dict[str, str] = {}
    for eid in example_ids:
        h = int.from_bytes(
            hashlib.sha256(groups[eid].encode()).digest()[:8], "little"
        ) / 2**64
        if h < test_frac:
            out[eid] = "test"
        elif h < test_frac + val_frac:
            out[eid] = "val"
        else:
            out[eid] = "train"
    return out


def substitute_placeholders(question: str, prior_answers: list[str]) -> str:
    """Replace '#k' references with the answer of decomposition step k."""

    def repl(m: re.Match) -> str:
        k = int(m.group(1)) - 1
        return prior_answers[k] if 0 <= k < len(prior_answers) else m.group(0)

    return _PLACEHOLDER.sub(repl, question)


def load_musique(split: str = "train"):
    """Load MuSiQue via HuggingFace datasets (both answerable and full sets
    include an `answerable` flag)."""
    from datasets import load_dataset

    return load_dataset(HF_DATASET, split=split)


def declarative(gateway: CachedGateway, model: str, question: str, answer: str) -> str:
    return gateway.chat(
        [{"role": "user", "content": prompts.DECLARATIVE_REWRITE_V1.format(
            question=question, answer=answer)}],
        model=model,
    ).strip()


def compose_bridge(gateway: CachedGateway, model: str, fact_a: str, fact_b: str) -> str:
    return gateway.chat(
        [{"role": "user", "content": prompts.COMPOSE_BRIDGE_V1.format(
            fact_a=fact_a, fact_b=fact_b)}],
        model=model,
    ).strip()


def extract_triples_for_example(
    example: dict,
    gateway: CachedGateway,
    model: str,
    group_id: str,
) -> list[Triple]:
    """Produce composition triples (possibly chained) for one MuSiQue example."""
    decomp = example["question_decomposition"]
    n_hops = len(decomp)
    if n_hops < 2 or n_hops > 4:
        return []
    answerable = bool(example.get("answerable", True))

    answers = [step["answer"] for step in decomp]
    hop_facts: list[str] = []
    for step_idx, step in enumerate(decomp):
        sub_q = substitute_placeholders(step["question"], answers[:step_idx])
        hop_facts.append(declarative(gateway, model, sub_q, step["answer"]))

    final_composed = declarative(gateway, model, example["question"], example["answer"])
    query = example["question"]
    triples: list[Triple] = []

    left = hop_facts[0]
    for k in range(1, n_hops):
        is_last = k == n_hops - 1
        composed = (
            final_composed
            if is_last
            else compose_bridge(gateway, model, left, hop_facts[k])
        )
        triples.append(
            Triple(
                qid=f"{example['id']}::pair{k - 1}",
                group_id=group_id,
                hops=n_hops,
                fact_a=left,
                fact_b=hop_facts[k],
                composed_gt=composed,
                query=query,
                answerable=answerable,
                pair_depth=k - 1,
            )
        )
        left = composed
    return triples


def build_triples(
    gateway: CachedGateway,
    model: str,
    out_dir: str | Path = "data/processed",
    max_examples: int | None = None,
    val_frac: float = 0.05,
    test_frac: float = 0.05,
    hf_split: str = "train",
) -> dict[str, int]:
    """End-to-end: load MuSiQue, split leakage-safely, rewrite with cached
    gateway calls, write musique_triples.{train,val,test}.jsonl.
    Unanswerable variants are written to musique_triples.unanswerable.jsonl."""
    ds = load_musique(hf_split)
    examples = list(ds) if max_examples is None else list(ds.select(range(max_examples)))
    split_map = split_by_group([ex["id"] for ex in examples], val_frac, test_frac)
    groups = assign_groups([ex["id"] for ex in examples])

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = {
        name: open(out / f"musique_triples.{name}.jsonl", "w")
        for name in ("train", "val", "test", "unanswerable")
    }
    counts = {name: 0 for name in files}
    try:
        for ex in examples:
            triples = extract_triples_for_example(
                ex, gateway, model, group_id=groups[ex["id"]]
            )
            for t in triples:
                dest = "unanswerable" if not t.answerable else split_map[ex["id"]]
                files[dest].write(json.dumps(asdict(t)) + "\n")
                counts[dest] += 1
    finally:
        for f in files.values():
            f.close()
    return counts


def read_triples(path: str | Path) -> list[Triple]:
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(Triple(**json.loads(line)))
    return rows
