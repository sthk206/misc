"""Pinned thresholds and models — tuning knobs, not constants (spec §8).

Every threshold used anywhere in the system lives here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

DATA_DIR = Path(os.environ.get("RAM_DATA_DIR", "data"))


@dataclass(frozen=True)
class Thresholds:
    ace_dedup_cosine: float = 0.85        # ADD -> INCR conversion (A.3)
    pattern_canon_cosine: float = 0.80    # pattern canonicalization gate (C.3)
    dod_coverage: float = 0.90            # attribution-coverage stopping rule (C.5)
    ace_token_budget: int = 2000          # playbook injection hard budget (A.2)
    ace_entry_max_tokens: int = 60        # per-entry directive length cap (A.1)
    escalation_depth_cap: int = 3         # investigator loop depth (C.5)
    pattern_hop_cap: int = 2              # pattern traversal hops (C.5)
    promotion_min_instances: int = 2      # pattern -> ACE candidate bar (C.7)
    promotion_severity_override: float = 8.0  # 1 instance suffices above this severity
    pattern_registry_soft_cap: int = 100  # past this, thresholds are wrong — re-tune


@dataclass(frozen=True)
class Models:
    embedder: str = "intfloat/e5-base-v2"   # local, also used by dedup/canonicalization
    judge: str = "claude-sonnet-5"          # pinned LLM judge for evals
    agent_model: str = "claude-sonnet-5"    # main agent sessions
    subagent_model: str = "claude-haiku-4-5-20251001"  # reflector / insight_writer


@dataclass(frozen=True)
class Paths:
    ace_db: Path = DATA_DIR / "ace.sqlite"
    prefs_db: Path = DATA_DIR / "prefs.sqlite"
    findings_db: Path = DATA_DIR / "findings.sqlite"
    episodes_dir: Path = DATA_DIR / "episodes"
    prefs_registry: Path = Path("configs/prefs_registry.yaml")
    denylists: Path = Path("configs/denylists.yaml")


@dataclass(frozen=True)
class Config:
    thresholds: Thresholds = field(default_factory=Thresholds)
    models: Models = field(default_factory=Models)
    paths: Paths = field(default_factory=Paths)


CONFIG = Config()
