"""Shared utilities: seeding, config loading, results writing."""

from __future__ import annotations

import json
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

RESULTS_DIR = Path(os.environ.get("LCM_RESULTS_DIR", "results"))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def git_hash() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "nogit"


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def write_results(
    name: str,
    config: dict[str, Any],
    metrics: dict[str, Any],
    extra: dict[str, Any] | None = None,
    results_dir: Path | None = None,
) -> Path:
    """Every run writes a JSON with git hash, config, metrics, token spend."""
    out_dir = results_dir or RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": name,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_hash": git_hash(),
        "config": config,
        "metrics": metrics,
    }
    if extra:
        payload.update(extra)
    out = out_dir / f"{name}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(payload, indent=2, default=str))
    return out


def token_jaccard(a: str, b: str) -> float:
    ta, tb = set(a.lower().split()), set(b.lower().split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)
