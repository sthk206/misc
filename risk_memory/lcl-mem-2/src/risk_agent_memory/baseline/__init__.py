"""Frozen baseline (spec 2.1): re-export the builder so eval code imports the
baseline from one place. Do not add stores here — a store ships only if it
beats THIS configuration through the same harness."""

from risk_agent_memory.agent.options import build_baseline_options

__all__ = ["build_baseline_options"]
