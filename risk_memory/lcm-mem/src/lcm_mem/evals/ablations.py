"""Phase 5 ablation grid: each toggle applied to the full system, same seeds.

Grid (from the plan):
  - pair scorer: predictor -> mean_pool / cross_encoder / llm_score / random
  - persistence ON/OFF (OFF: derived facts discarded after each query)
  - query-conditioning of the predictor ON/OFF (checkpoint choice)
  - confidence decay ON/OFF
  - fine-tuned encoder vs stock e5
Invalidation ON/OFF is exercised in the LongMemEval knowledge-update slice by
disabling contradiction checks at ingest time.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from lcm_mem.common import write_results
from lcm_mem.llm.gateway import CachedGateway
from lcm_mem.memory.compose import ComposeConfig


def ablation_grid(base: ComposeConfig, ckpt_noq: str | None = None) -> dict[str, ComposeConfig]:
    grid: dict[str, ComposeConfig] = {"full": base}
    for scorer in ("mean_pool", "cross_encoder", "llm_score", "random"):
        grid[f"scorer={scorer}"] = replace(base, scorer=scorer)
    grid["no_decay"] = replace(base, confidence_decay=1.0)
    if ckpt_noq:
        grid["no_query_conditioning"] = replace(base, ckpt_path=ckpt_noq)
    return grid


def run_ablations(
    gateway: CachedGateway,
    base: ComposeConfig,
    runner,                      # callable(gateway, cfg, **kw) -> metrics dict
    ckpt_noq: str | None = None,
    results_dir: str | Path = "results",
    **runner_kwargs,
) -> dict[str, dict]:
    """Run `runner` (e.g. run_longmemeval) once per grid cell."""
    out: dict[str, dict] = {}
    for name, cfg in ablation_grid(base, ckpt_noq).items():
        out[name] = runner(gateway, cfg, **runner_kwargs)
    write_results(
        "ablation_grid",
        config={"base": base.__dict__, "cells": list(out)},
        metrics=out,
        results_dir=Path(results_dir),
    )
    return out
