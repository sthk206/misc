"""CLI entry point: every experiment is a command taking a config path.

Usage:
  lcm build-triples configs/data.yaml
  lcm phase1 configs/phase1.yaml
  lcm train configs/phase1.yaml --predictor mlp --loss cosine
  lcm longmemeval configs/longmemeval.yaml
  lcm cache-stats
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from lcm_mem.common import load_config, write_results

app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)
console = Console()


def _gateway(cfg: dict):
    from lcm_mem.llm.gateway import CachedGateway

    return CachedGateway(
        cache_path=cfg.get("llm_cache", "cache/llm_cache.sqlite"),
        dry_run=bool(cfg.get("dry_run", False)),
        max_calls=cfg.get("max_llm_calls_budget"),
    )


@app.command()
def build_triples(config: Path):
    """Phase 0: MuSiQue -> composition triples (train/val/test jsonl)."""
    cfg = load_config(config)
    from lcm_mem.data.musique import build_triples as _build

    gw = _gateway(cfg)
    counts = _build(
        gateway=gw,
        model=cfg["rewrite_model"],
        out_dir=cfg.get("out_dir", "data/processed"),
        max_examples=cfg.get("max_examples"),
        val_frac=cfg.get("val_frac", 0.05),
        test_frac=cfg.get("test_frac", 0.05),
    )
    console.print(counts)
    console.print(gw.usage.as_dict())
    write_results("build_triples", cfg, {"counts": counts, "usage": gw.usage.as_dict()})


@app.command()
def audit_triples(config: Path, n: int = 50):
    """Print n random triples for the manual rewrite-quality audit."""
    import random

    cfg = load_config(config)
    from lcm_mem.data.musique import read_triples

    triples = read_triples(Path(cfg.get("out_dir", "data/processed")) / "musique_triples.train.jsonl")
    for t in random.Random(0).sample(triples, min(n, len(triples))):
        console.print(f"[bold]{t.qid}[/bold] ({t.hops}-hop)")
        console.print(f"  A: {t.fact_a}\n  B: {t.fact_b}\n  =>: {t.composed_gt}\n  Q: {t.query}\n")


@app.command()
def phase1(config: Path):
    """Phase 1 kill experiment: train all predictors, evaluate, write verdict."""
    cfg = load_config(config)
    from lcm_mem.evals.run_phase1 import run_phase1

    result = run_phase1(
        data_dir=cfg.get("data_dir", "data/processed"),
        emb_dir=cfg.get("emb_dir", "data/embeddings"),
        ckpt_dir=cfg.get("ckpt_dir", "models"),
        embedder_name=cfg.get("embedder", "intfloat/e5-base-v2"),
        methods=tuple(cfg.get("methods", ["mean_pool", "learned_pool", "mlp", "attn"])),
        losses=tuple(cfg.get("losses", ["cosine", "infonce"])),
        query_ablation=cfg.get("query_ablation", True),
        include_cross_encoder=cfg.get("include_cross_encoder", True),
        n_pruning_items=cfg.get("n_pruning_items", 200),
        seed=cfg.get("seed", 0),
        train_overrides={
            k: cfg[k] for k in ("lr", "batch_size", "max_epochs", "patience") if k in cfg
        },
    )
    console.print(f"[bold]{result['verdict']}[/bold]\n{result['rationale']}")


@app.command()
def train(
    config: Path,
    predictor: str = "mlp",
    loss: str = "cosine",
    use_query: bool = True,
):
    """Train a single predictor variant (for iteration outside the full grid)."""
    cfg = load_config(config)
    from lcm_mem.predictor.train import TrainConfig, train_predictor

    tc = TrainConfig(
        predictor=predictor, loss=loss, use_query=use_query,
        seed=cfg.get("seed", 0),
        lr=cfg.get("lr", 1e-4), batch_size=cfg.get("batch_size", 256),
    )
    emb_dir = Path(cfg.get("emb_dir", "data/embeddings"))
    name = f"phase1_{predictor}_{loss}{'' if use_query else '_noq'}"
    out = train_predictor(
        emb_dir / "train.npz", emb_dir / "val.npz", tc,
        ckpt_path=Path(cfg.get("ckpt_dir", "models")) / f"{name}.pt",
    )
    out.pop("model")
    console.print({k: v for k, v in out.items() if k != "history"})
    write_results(f"train_{name}", cfg, {k: v for k, v in out.items() if k != "history"})


@app.command()
def longmemeval(config: Path):
    """Phase 5: LongMemEval-S end-to-end run."""
    cfg = load_config(config)
    from lcm_mem.evals.longmemeval_runner import run_longmemeval
    from lcm_mem.memory.compose import ComposeConfig

    gw = _gateway(cfg)
    compose_cfg = ComposeConfig(**cfg.get("compose", {}))
    metrics = run_longmemeval(
        gw, compose_cfg,
        data_path=cfg.get("data_path"),
        embedder_name=cfg.get("embedder", "intfloat/e5-base-v2"),
        extract_model=cfg.get("extract_model", compose_cfg.model),
        judge_model=cfg.get("judge_model", compose_cfg.model),
        limit=cfg.get("limit"),
    )
    console.print(metrics)


@app.command()
def cache_stats():
    """Show LLM cache size and entry count."""
    import sqlite3

    p = Path("cache/llm_cache.sqlite")
    if not p.exists():
        console.print("no cache yet")
        raise typer.Exit()
    db = sqlite3.connect(p)
    n, pt, ct = db.execute(
        "SELECT COUNT(*), COALESCE(SUM(prompt_tokens),0), COALESCE(SUM(completion_tokens),0) FROM llm_cache"
    ).fetchone()
    console.print(f"{n} cached responses, {pt} prompt tokens, {ct} completion tokens, "
                  f"{p.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    app()
