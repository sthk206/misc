"""Phase 1 orchestration: embed triples -> train all predictors -> run the full
kill-experiment evaluation -> write results JSON + phase1_verdict.md.

Every step reads/writes cached artifacts so reruns are cheap.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

from lcm_mem.common import set_seed, write_results
from lcm_mem.data.hard_negatives import build_entity_pool, generate_hard_negatives
from lcm_mem.data.musique import Triple, read_triples
from lcm_mem.encoder.embed import BaseEmbedder, get_embedder, pick_device
from lcm_mem.evals.kill_experiment import (
    PruningItem,
    VerdictInputs,
    decide_verdict,
    discrimination_eval,
    make_pair_scorer_fn,
    overlap_terciles,
    pair_pruning_eval,
    retrieval_eval,
    write_verdict,
)
from lcm_mem.predictor.train import TrainConfig, train_predictor


def embed_triples_npz(
    triples: list[Triple],
    embedder: BaseEmbedder,
    out_path: str | Path,
    batch_size: int = 64,
) -> Path:
    out_path = Path(out_path)
    if out_path.exists():
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    a = embedder.encode([t.fact_a for t in triples], kind="passage", batch_size=batch_size)
    b = embedder.encode([t.fact_b for t in triples], kind="passage", batch_size=batch_size)
    q = embedder.encode([t.query for t in triples], kind="query", batch_size=batch_size)
    tgt = embedder.encode([t.composed_gt for t in triples], kind="passage", batch_size=batch_size)
    np.savez(out_path, a=a, b=b, q=q, target=tgt)
    return out_path


def _predict(model, npz_path: Path, device: str, use_query: bool) -> np.ndarray:
    z = np.load(npz_path)
    with torch.no_grad():
        a = torch.from_numpy(z["a"]).float().to(device)
        b = torch.from_numpy(z["b"]).float().to(device)
        q = torch.from_numpy(z["q"]).float().to(device)
        pred = model(a, b, q if use_query else None)
    return pred.cpu().numpy()


def build_distractor_pool(
    test_triples: list[Triple], embedder: BaseEmbedder, seed: int = 0,
    corrupt_frac: float = 0.3,
) -> np.ndarray:
    """Distractors = all other composed facts + corrupted variants of a subset."""
    rng = random.Random(seed)
    texts = [t.composed_gt for t in test_triples]
    pool = build_entity_pool(texts)
    corrupted: list[str] = []
    for t in rng.sample(test_triples, int(len(test_triples) * corrupt_frac)):
        corrupted.extend(n.text for n in generate_hard_negatives(t.composed_gt, pool, rng))
    return embedder.encode(texts + corrupted, kind="passage")


def build_pruning_items(
    test_triples: list[Triple], n_items: int = 200, n_facts: int = 20, seed: int = 0
) -> list[PruningItem]:
    """Each item: a question's two gold facts + distractor facts drawn from
    other items (stand-ins for distractor-paragraph facts)."""
    rng = random.Random(seed)
    all_facts = [f for t in test_triples for f in (t.fact_a, t.fact_b)]
    items = []
    for t in rng.sample(test_triples, min(n_items, len(test_triples))):
        distractors = []
        while len(distractors) < n_facts - 2:
            f = rng.choice(all_facts)
            if f not in (t.fact_a, t.fact_b) and f not in distractors:
                distractors.append(f)
        facts = distractors + [t.fact_a, t.fact_b]
        rng.shuffle(facts)
        items.append(
            PruningItem(
                query=t.query,
                facts=facts,
                gold_pair=(facts.index(t.fact_a), facts.index(t.fact_b)),
            )
        )
    return items


def run_phase1(
    data_dir: str | Path = "data/processed",
    emb_dir: str | Path = "data/embeddings",
    ckpt_dir: str | Path = "models",
    embedder_name: str = "intfloat/e5-base-v2",
    methods: tuple[str, ...] = ("mean_pool", "learned_pool", "mlp", "attn"),
    losses: tuple[str, ...] = ("cosine", "infonce"),
    query_ablation: bool = True,
    include_cross_encoder: bool = True,
    n_pruning_items: int = 200,
    seed: int = 0,
    results_dir: str | Path = "results",
    train_overrides: dict | None = None,
) -> dict:
    set_seed(seed)
    device = pick_device()
    embedder = get_embedder(embedder_name)
    data_dir, emb_dir, ckpt_dir = Path(data_dir), Path(emb_dir), Path(ckpt_dir)

    splits = {
        name: read_triples(data_dir / f"musique_triples.{name}.jsonl")
        for name in ("train", "val", "test")
    }
    npz = {
        name: embed_triples_npz(ts, embedder, emb_dir / f"{name}.npz")
        for name, ts in splits.items()
    }
    test_triples = splits["test"]
    terciles = overlap_terciles(test_triples)
    targets = np.load(npz["test"])["target"]
    distractors = build_distractor_pool(test_triples, embedder, seed)
    pruning_items = build_pruning_items(test_triples, n_pruning_items, seed=seed)

    all_metrics: dict[str, dict] = {}
    mrr_overall: dict[str, float] = {}
    mrr_low: dict[str, float] = {}
    pruning: dict[str, dict] = {}
    discrimination: dict[str, dict] = {}

    overrides = train_overrides or {}

    def _cfg(**kw) -> TrainConfig:
        return TrainConfig(**{**kw, **overrides, "seed": seed})

    variants: list[tuple[str, TrainConfig]] = []
    for m in methods:
        if m in ("mean_pool", "learned_pool"):
            variants.append((m, _cfg(predictor=m, loss="cosine")))
            continue
        for loss in losses:
            variants.append((f"{m}_{loss}", _cfg(predictor=m, loss=loss)))
            if query_ablation:
                variants.append(
                    (f"{m}_{loss}_noq", _cfg(predictor=m, loss=loss, use_query=False))
                )

    best_learned: tuple[str, float] | None = None
    for name, cfg in variants:
        ckpt = ckpt_dir / f"phase1_{name}.pt"
        out = train_predictor(npz["train"], npz["val"], cfg, ckpt_path=ckpt)
        model = out.pop("model")
        dev = cfg.device or device
        preds = _predict(model, npz["test"], dev, cfg.use_query)
        rm = retrieval_eval(preds, targets, distractors, seed=seed)
        per_tercile = {}
        for terc in (0, 1, 2):
            mask = terciles == terc
            if mask.sum() == 0:
                continue
            per_tercile[terc] = retrieval_eval(
                preds[mask], targets[mask], distractors, seed=seed
            ).as_dict()
        disc = discrimination_eval(test_triples, preds, embedder, seed=seed)
        all_metrics[name] = {
            "retrieval": rm.as_dict(),
            "per_tercile": per_tercile,
            "discrimination": disc,
            "train": {k: v for k, v in out.items() if k != "history"},
        }
        key = "mean_pool" if name == "mean_pool" else name
        mrr_overall[key] = rm.mrr
        mrr_low[key] = per_tercile.get(0, {}).get("MRR", 0.0)
        discrimination[key] = disc
        # track the best learned predictor for the verdict + pruning eval
        if name not in ("mean_pool", "learned_pool"):
            if best_learned is None or rm.mrr > best_learned[1]:
                best_learned = (name, rm.mrr)
                pruning["predictor"] = pair_pruning_eval(
                    pruning_items,
                    make_pair_scorer_fn("predictor", embedder, model=model, device=dev),
                )
                mrr_overall["predictor"] = rm.mrr
                mrr_low["predictor"] = mrr_low[key]

    pruning["mean_pool"] = pair_pruning_eval(
        pruning_items, make_pair_scorer_fn("mean_pool", embedder)
    )
    if include_cross_encoder:
        pruning["cross_encoder"] = pair_pruning_eval(
            pruning_items, make_pair_scorer_fn("cross_encoder", embedder)
        )

    verdict, rationale = decide_verdict(
        VerdictInputs(
            mrr_overall=mrr_overall,
            mrr_low_overlap=mrr_low,
            pruning=pruning,
            discrimination=discrimination,
        )
    )
    write_verdict(verdict, rationale, str(Path(results_dir) / "phase1_verdict.md"))
    result = {
        "verdict": verdict,
        "rationale": rationale,
        "best_learned": best_learned[0] if best_learned else None,
        "per_method": all_metrics,
        "pruning": pruning,
    }
    write_results(
        "phase1_kill_experiment",
        config={
            "embedder": embedder_name, "methods": list(methods),
            "losses": list(losses), "seed": seed,
        },
        metrics=result,
        results_dir=Path(results_dir),
    )
    return result
