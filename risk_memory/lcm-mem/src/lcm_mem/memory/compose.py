"""Query-time best-first composition search — the core loop.

1. Embed query; extract query entities (cached gateway call).
2. Candidates = union of top-k dense retrieval and entity-linked facts.
3. Answerability check: cheap similarity heuristic first, else one gateway call.
4. Best-first search: score all frontier pairs with the configured scorer,
   pop the best pair, verbalize the composition with ONE gateway call, persist
   non-NONE inferences as derived facts (with provenance), add to frontier.
5. Stop on: answerable, LLM budget exhausted, max depth, or all scores below
   threshold. Answer from the final set with citations to fact ids.

Persistence is the point: derived facts survive across queries. We log the
fraction of queries answered using previously derived facts and LLM calls
saved.
"""

from __future__ import annotations

import heapq
import itertools
import random
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import torch

from lcm_mem.encoder.embed import l2_normalize
from lcm_mem.llm import prompts
from lcm_mem.llm.gateway import CachedGateway
from lcm_mem.memory.store import MemoryStore


# ---------------------------------------------------------------- scorers

class PairScorer(Protocol):
    def score(
        self,
        emb_a: np.ndarray, emb_b: np.ndarray, emb_q: np.ndarray,
        text_a: str, text_b: str, query: str,
    ) -> float: ...


class MeanPoolScorer:
    """cos(normalize((a+b)/2), q)"""

    def score(self, emb_a, emb_b, emb_q, text_a, text_b, query) -> float:
        pooled = l2_normalize((emb_a + emb_b) / 2)
        return float(np.dot(pooled, emb_q))


class PredictorScorer:
    """cos(f(a, b, q), q) with the trained latent predictor."""

    def __init__(self, ckpt_path: str, device: str | None = None):
        from lcm_mem.encoder.embed import pick_device
        from lcm_mem.predictor.train import load_predictor

        self.device = device or pick_device()
        self.model = load_predictor(ckpt_path, self.device)

    def score(self, emb_a, emb_b, emb_q, text_a, text_b, query) -> float:
        with torch.no_grad():
            a = torch.from_numpy(emb_a).float().unsqueeze(0).to(self.device)
            b = torch.from_numpy(emb_b).float().unsqueeze(0).to(self.device)
            q = torch.from_numpy(emb_q).float().unsqueeze(0).to(self.device)
            pred = self.model(a, b, q if getattr(self.model, "use_query", True) else None)
            return float((pred[0] @ q[0]).item())


class CrossEncoderScorer:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from lcm_mem.predictor.baselines import CrossEncoderBaseline

        self.ce = CrossEncoderBaseline(model_name)

    def score(self, emb_a, emb_b, emb_q, text_a, text_b, query) -> float:
        return float(self.ce.score_pairs(query, [(text_a, text_b)])[0])


class LLMScorer:
    def __init__(self, gateway: CachedGateway, model: str):
        self.gateway, self.model = gateway, model

    def score(self, emb_a, emb_b, emb_q, text_a, text_b, query) -> float:
        out = self.gateway.chat(
            [{"role": "user", "content": prompts.PAIR_SCORE_V1.format(
                fact_a=text_a, fact_b=text_b, query=query)}],
            model=self.model,
        ).strip()
        try:
            return float(out.split()[0]) / 10.0
        except (ValueError, IndexError):
            return 0.0


class RandomScorer:
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def score(self, emb_a, emb_b, emb_q, text_a, text_b, query) -> float:
        return self.rng.random()


def build_scorer(
    name: str,
    gateway: CachedGateway | None = None,
    model: str | None = None,
    ckpt_path: str | None = None,
    seed: int = 0,
) -> PairScorer:
    if name == "predictor":
        assert ckpt_path, "predictor scorer needs ckpt_path"
        return PredictorScorer(ckpt_path)
    if name == "mean_pool":
        return MeanPoolScorer()
    if name == "cross_encoder":
        return CrossEncoderScorer()
    if name == "llm_score":
        assert gateway is not None and model is not None
        return LLMScorer(gateway, model)
    if name == "random":
        return RandomScorer(seed)
    raise ValueError(f"unknown scorer {name}")


# ---------------------------------------------------------------- config

@dataclass
class ComposeConfig:
    scorer: str = "predictor"     # predictor | mean_pool | cross_encoder | llm_score | random
    ckpt_path: str | None = None
    model: str = "gpt-4o-mini"    # config-driven, never hardcode elsewhere
    k_dense: int = 20
    max_llm_calls: int = 5        # composition-call budget L
    max_depth: int = 3            # D
    t_answerable: float = 0.82    # cheap heuristic threshold
    min_pair_score: float = 0.1
    confidence_decay: float = 0.85
    seed: int = 0


@dataclass
class ComposeResult:
    answer: str
    citations: list[int]
    derived_ids: list[int] = field(default_factory=list)
    reused_derived_ids: list[int] = field(default_factory=list)
    llm_calls: int = 0
    stopped_because: str = ""


# ---------------------------------------------------------------- loop

def extract_query_entities(gateway: CachedGateway, model: str, query: str) -> list[str]:
    try:
        out = gateway.chat_json(
            [{"role": "user", "content": prompts.QUERY_ENTITIES_V1.format(query=query)}],
            model=model,
        )
        return [str(e) for e in out]
    except Exception:  # noqa: BLE001 — entity extraction is best-effort
        return []


def _answerable_heuristic(
    store: MemoryStore, fact_ids: list[int], emb_q: np.ndarray, t: float
) -> bool:
    return any(float(np.dot(store.embedding(f), emb_q)) >= t for f in fact_ids)


def _answerable_llm(
    gateway: CachedGateway, model: str, store: MemoryStore,
    fact_ids: list[int], query: str,
) -> bool:
    facts = "\n".join(f"- {store.get_fact(f).text}" for f in fact_ids)
    out = gateway.chat(
        [{"role": "user", "content": prompts.ANSWERABILITY_V1.format(
            facts=facts, question=query)}],
        model=model,
    ).strip().lower()
    return out.startswith("yes")


def _final_answer(
    gateway: CachedGateway, model: str, store: MemoryStore,
    fact_ids: list[int], query: str,
) -> str:
    facts = "\n".join(
        f"{fid}. {store.get_fact(fid).text}" for fid in fact_ids
    )
    return gateway.chat(
        [{"role": "user", "content": prompts.FINAL_ANSWER_V1.format(
            facts=facts, question=query)}],
        model=model,
    ).strip()


def answer_query(
    store: MemoryStore,
    query: str,
    gateway: CachedGateway,
    cfg: ComposeConfig,
    scorer: PairScorer | None = None,
) -> ComposeResult:
    scorer = scorer or build_scorer(
        cfg.scorer, gateway=gateway, model=cfg.model,
        ckpt_path=cfg.ckpt_path, seed=cfg.seed,
    )
    emb_q = store.embedder.encode([query], kind="query")[0]
    llm_calls = 0  # composition/answerability budget; final answer not counted

    # candidates: dense top-k ∪ entity-linked
    dense = [fid for fid, _ in store.search(query, k=cfg.k_dense)]
    q_entities = extract_query_entities(gateway, cfg.model, query)
    entity_linked = store.facts_with_entities(q_entities)
    frontier: list[int] = sorted(set(dense) | set(entity_linked))
    reused = [
        f for f in frontier if store.get_fact(f).type in ("derived", "world_bridge")
    ]

    def answerable() -> bool:
        nonlocal llm_calls
        if _answerable_heuristic(store, frontier, emb_q, cfg.t_answerable):
            return True
        if llm_calls >= cfg.max_llm_calls or not frontier:
            return False
        llm_calls += 1
        return _answerable_llm(gateway, cfg.model, store, frontier, query)

    stopped = "answerable"
    result_derived: list[int] = []
    if not answerable():
        # -- best-first search over compositions --
        heap: list[tuple[float, int, int, int]] = []  # (-score, tiebreak, a, b)
        scored: set[tuple[int, int]] = set()
        counter = itertools.count()

        def push_pairs(new_ids: list[int]) -> None:
            for a in frontier:
                for b in new_ids:
                    key = (min(a, b), max(a, b))
                    if a == b or key in scored:
                        continue
                    scored.add(key)
                    s = scorer.score(
                        store.embedding(key[0]), store.embedding(key[1]), emb_q,
                        store.get_fact(key[0]).text, store.get_fact(key[1]).text,
                        query,
                    )
                    heapq.heappush(heap, (-s, next(counter), key[0], key[1]))

        push_pairs(list(frontier))
        stopped = "exhausted_pairs"
        derived_ids: list[int] = []
        while heap:
            neg_s, _, a, b = heapq.heappop(heap)
            if -neg_s < cfg.min_pair_score:
                stopped = "below_threshold"
                break
            if llm_calls >= cfg.max_llm_calls:
                stopped = "budget"
                break
            depth = max(store.get_fact(a).depth, store.get_fact(b).depth) + 1
            if depth > cfg.max_depth:
                continue
            llm_calls += 1
            out = gateway.chat_json(
                [{"role": "user", "content": prompts.COMPOSE_INFERENCE_V1.format(
                    fact_a=store.get_fact(a).text, fact_b=store.get_fact(b).text,
                    query=query)}],
                model=cfg.model,
            )
            inference = str(out.get("inference", "NONE")).strip()
            if inference.upper() == "NONE" or not inference:
                continue
            llm_conf = float(out.get("confidence", 0.5))
            world = bool(out.get("used_world_knowledge", False))
            conf = llm_conf * (cfg.confidence_decay ** depth)
            fid = store.add_fact(
                text=inference,
                type="world_bridge" if world else "derived",
                confidence=conf,
                parents=[a, b],
                entities=list(set(store.entities_of(a)) | set(store.entities_of(b))),
                extraction_model=cfg.model,
            )
            derived_ids.append(fid)
            push_pairs([fid])
            frontier.append(fid)
            if _answerable_heuristic(store, [fid], emb_q, cfg.t_answerable):
                stopped = "answerable"
                break
        result_derived = derived_ids

    answer = (
        _final_answer(gateway, cfg.model, store, frontier, query)
        if frontier
        else "I don't know"
    )
    citations = _parse_citations(answer, frontier)
    return ComposeResult(
        answer=answer,
        citations=citations,
        derived_ids=result_derived,
        reused_derived_ids=reused,
        llm_calls=llm_calls,
        stopped_because=stopped or "answerable",
    )


def _parse_citations(answer: str, valid_ids: list[int]) -> list[int]:
    import re

    cited = [int(m) for m in re.findall(r"\[(\d+)\]", answer)]
    valid = set(valid_ids)
    return [c for c in cited if c in valid]
