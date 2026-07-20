"""Phase 1 KILL EXPERIMENT: can composition be predicted in latent space?

Implements exactly the evaluation protocol from the plan:
  1. R@1, R@10, MRR of retrieving composed_gt from a pool of 1000
     (true target + 999 distractors: other composed facts + corrupted variants).
  2. Lexical-overlap stratification into terciles of
     max token-Jaccard(composed_gt, fact_a or fact_b); the low-overlap tercile
     is THE KEY CELL.
  3. Discrimination test: rank composed_gt above its own date-swapped, negated,
     entity-swapped variants.
  4. Downstream proxy — pair pruning: recall of the gold pair in top-k scored
     pairs among 20 candidate facts, plus latency per 1000 pairs.

Decision gate written to results/phase1_verdict.md:
  STRONG PASS / WEAK PASS (filter regime) / FAIL.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from lcm_mem.common import token_jaccard
from lcm_mem.data.hard_negatives import generate_hard_negatives, build_entity_pool
from lcm_mem.data.musique import Triple
from lcm_mem.encoder.embed import BaseEmbedder, l2_normalize


# ------------------------------------------------------------ retrieval

@dataclass
class RetrievalMetrics:
    r_at_1: float
    r_at_10: float
    mrr: float
    n: int

    def as_dict(self) -> dict:
        return {"R@1": self.r_at_1, "R@10": self.r_at_10, "MRR": self.mrr, "n": self.n}


def retrieval_eval(
    preds: np.ndarray,          # (N, d) L2-normalized predictions
    targets: np.ndarray,        # (N, d) L2-normalized composed_gt embeddings
    distractors: np.ndarray,    # (M, d) pool of distractor embeddings
    pool_size: int = 1000,
    seed: int = 0,
) -> RetrievalMetrics:
    """Per item: pool = own target + (pool_size - 1) sampled distractors
    (other targets + corrupted variants)."""
    rng = np.random.default_rng(seed)
    n = preds.shape[0]
    r1 = r10 = 0
    rr_sum = 0.0
    n_distract = min(pool_size - 1, distractors.shape[0])
    for i in range(n):
        idx = rng.choice(distractors.shape[0], size=n_distract, replace=False)
        pool = np.vstack([targets[i : i + 1], distractors[idx]])
        sims = pool @ preds[i]
        rank = int((sims > sims[0]).sum()) + 1
        rr_sum += 1.0 / rank
        r1 += rank == 1
        r10 += rank <= 10
    return RetrievalMetrics(r1 / n, r10 / n, rr_sum / n, n)


def overlap_terciles(triples: list[Triple]) -> np.ndarray:
    """Bucket items 0/1/2 by max token-Jaccard(composed_gt, fact_a|fact_b).
    Bucket 0 is the low-overlap tercile — the key cell."""
    overlaps = np.array(
        [
            max(token_jaccard(t.composed_gt, t.fact_a), token_jaccard(t.composed_gt, t.fact_b))
            for t in triples
        ]
    )
    lo, hi = np.quantile(overlaps, [1 / 3, 2 / 3])
    return np.digitize(overlaps, [lo, hi])  # 0 = low overlap


# ------------------------------------------------------------ discrimination

def discrimination_eval(
    triples: list[Triple],
    preds: np.ndarray,
    embedder: BaseEmbedder,
    seed: int = 0,
) -> dict:
    """For each item build the pool {composed_gt + its corrupted variants} and
    check the true one ranks first under the prediction. Measures whether the
    latent space encodes correctness-relevant detail at all."""
    rng = random.Random(seed)
    pool_texts = build_entity_pool([t.composed_gt for t in triples])
    correct = 0
    n_used = 0
    per_kind_wrong: dict[str, int] = {}
    for i, t in enumerate(triples):
        negs = generate_hard_negatives(t.composed_gt, pool_texts, rng)
        if not negs:
            continue
        texts = [t.composed_gt] + [n.text for n in negs]
        embs = embedder.encode(texts, kind="passage")
        sims = embs @ preds[i]
        n_used += 1
        if int(np.argmax(sims)) == 0:
            correct += 1
        else:
            kind = negs[int(np.argmax(sims)) - 1].kind
            per_kind_wrong[kind] = per_kind_wrong.get(kind, 0) + 1
    return {
        "accuracy": correct / n_used if n_used else float("nan"),
        "n": n_used,
        "confused_by": per_kind_wrong,
    }


# ------------------------------------------------------------ pair pruning

@dataclass
class PruningItem:
    """One held-out question with candidate facts; exactly one gold pair."""

    query: str
    facts: list[str]           # 20 candidates (gold supporting + distractors)
    gold_pair: tuple[int, int]  # indices into facts


def pair_pruning_eval(
    items: list[PruningItem],
    score_pairs_fn,            # (query, facts, pair_indices) -> np.ndarray of scores
    ks: tuple[int, ...] = (1, 3, 5, 10),
) -> dict:
    """Recall of the gold pair among top-k scored pairs + latency/1000 pairs."""
    recalls = {k: 0 for k in ks}
    total_pairs = 0
    t0 = time.perf_counter()
    for item in items:
        n = len(item.facts)
        pair_idx = [(i, j) for i in range(n) for j in range(i + 1, n)]
        total_pairs += len(pair_idx)
        scores = np.asarray(score_pairs_fn(item.query, item.facts, pair_idx))
        order = np.argsort(-scores)
        gold = tuple(sorted(item.gold_pair))
        rank = next(
            (r for r, oi in enumerate(order) if tuple(sorted(pair_idx[oi])) == gold),
            len(pair_idx),
        )
        for k in ks:
            recalls[k] += rank < k
    elapsed = time.perf_counter() - t0
    n_items = max(len(items), 1)
    return {
        **{f"gold_recall@{k}": recalls[k] / n_items for k in ks},
        "latency_ms_per_1000_pairs": 1000 * elapsed / max(total_pairs, 1) * 1000,
        "n_items": len(items),
        "n_pairs": total_pairs,
    }


def make_pair_scorer_fn(method: str, embedder: BaseEmbedder, model=None, device: str = "cpu"):
    """Batch pair scorer for the pruning eval.

    method: 'predictor' (needs model), 'mean_pool', 'cross_encoder'.
    Predictor/mean-pool priority = cos(pred, emb_q).
    """
    if method == "cross_encoder":
        from lcm_mem.predictor.baselines import CrossEncoderBaseline

        ce = CrossEncoderBaseline()

        def fn(query: str, facts: list[str], pair_idx: list[tuple[int, int]]):
            return ce.score_pairs(query, [(facts[i], facts[j]) for i, j in pair_idx])

        return fn

    def fn(query: str, facts: list[str], pair_idx: list[tuple[int, int]]):
        embs = embedder.encode(facts, kind="passage")
        q = embedder.encode([query], kind="query")[0]
        a = np.stack([embs[i] for i, _ in pair_idx])
        b = np.stack([embs[j] for _, j in pair_idx])
        if method == "mean_pool":
            pooled = l2_normalize((a + b) / 2)
            return pooled @ q
        if method == "predictor":
            with torch.no_grad():
                ta = torch.from_numpy(a).float().to(device)
                tb = torch.from_numpy(b).float().to(device)
                tq = torch.from_numpy(np.tile(q, (a.shape[0], 1))).float().to(device)
                pred = model(ta, tb, tq if getattr(model, "use_query", True) else None)
                return (pred @ torch.from_numpy(q).float().to(device)).cpu().numpy()
        raise ValueError(method)

    return fn


# ------------------------------------------------------------ verdict

@dataclass
class VerdictInputs:
    mrr_overall: dict[str, float]          # method -> MRR
    mrr_low_overlap: dict[str, float]      # method -> MRR in low-overlap tercile
    pruning: dict[str, dict]               # method -> pair_pruning_eval output
    discrimination: dict[str, dict] = field(default_factory=dict)


def decide_verdict(v: VerdictInputs) -> tuple[str, str]:
    """Returns (verdict, rationale). Gates from the plan:

    STRONG: predictor beats MeanPool by >= 10 MRR points overall AND in the
      low-overlap tercile AND matches/beats cross-encoder recall@5 for pruning
      at >= 10x lower latency.
    WEAK  : predictor gold-pair recall@10 >= 0.90 while pruning >= 90% of pairs.
    FAIL  : neither.
    """
    lines = []
    pred_mrr = v.mrr_overall.get("predictor", 0.0)
    mean_mrr = v.mrr_overall.get("mean_pool", 0.0)
    pred_lo = v.mrr_low_overlap.get("predictor", 0.0)
    mean_lo = v.mrr_low_overlap.get("mean_pool", 0.0)
    gain_overall = (pred_mrr - mean_mrr) * 100
    gain_lo = (pred_lo - mean_lo) * 100
    lines.append(f"MRR gain over MeanPool: overall {gain_overall:+.1f} pts, "
                 f"low-overlap tercile {gain_lo:+.1f} pts")

    strong = gain_overall >= 10 and gain_lo >= 10
    pr = v.pruning.get("predictor", {})
    ce = v.pruning.get("cross_encoder", {})
    if pr and ce:
        rec_ok = pr.get("gold_recall@5", 0) >= ce.get("gold_recall@5", 1)
        lat_ok = (
            ce.get("latency_ms_per_1000_pairs", 0)
            >= 10 * pr.get("latency_ms_per_1000_pairs", float("inf"))
        )
        lines.append(
            f"pruning recall@5: predictor {pr.get('gold_recall@5', 0):.3f} vs "
            f"cross-encoder {ce.get('gold_recall@5', 0):.3f}; latency/1000 pairs: "
            f"{pr.get('latency_ms_per_1000_pairs', 0):.1f}ms vs "
            f"{ce.get('latency_ms_per_1000_pairs', 0):.1f}ms"
        )
        strong = strong and rec_ok and lat_ok

    if strong:
        return "STRONG PASS", "\n".join(lines)

    # weak pass: high-recall filter regime. top-10 of C(20,2)=190 pairs prunes ~95%.
    if pr:
        n_pairs_per_item = pr.get("n_pairs", 0) / max(pr.get("n_items", 1), 1)
        prune_frac = 1 - 10 / n_pairs_per_item if n_pairs_per_item else 0.0
        r10 = pr.get("gold_recall@10", 0.0)
        lines.append(f"filter regime: gold recall@10 {r10:.3f}, prunes {prune_frac:.0%} of pairs")
        if r10 >= 0.90 and prune_frac >= 0.90:
            return "WEAK PASS", "\n".join(lines)
    return "FAIL", "\n".join(lines)


def write_verdict(verdict: str, rationale: str, path: str = "results/phase1_verdict.md") -> None:
    from pathlib import Path

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    consequences = {
        "STRONG PASS": "Proceed with the full plan; predictor is the pair scorer.",
        "WEAK PASS": ("Proceed; position the predictor as a high-recall filter and "
                       "let the cross-encoder rerank the survivors."),
        "FAIL": ("Proceed with Phases 3-5 using cross-encoder or LLM-scored pruning; "
                  "the paper pivots to provenance/invalidation plus an analysis of why "
                  "latent composition fails (see discrimination-test numbers)."),
    }
    p.write_text(
        f"# Phase 1 verdict: {verdict}\n\n{rationale}\n\n"
        f"**Consequence:** {consequences[verdict]}\n"
    )
