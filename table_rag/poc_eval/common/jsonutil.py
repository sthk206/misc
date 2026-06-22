"""Tiny shared helper: best-effort extraction of a JSON object from an LLM reply."""

from __future__ import annotations

import json
import re
from typing import Any


def extract_json(raw: str) -> dict[str, Any]:
    """Return the first JSON object found in `raw`, or {} if none parses.
    Tolerates ```json fences and surrounding prose."""
    if not raw:
        return {}
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    candidates = []
    if fenced:
        candidates.append(fenced.group(1))
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        candidates.append(m.group(0))
    for c in candidates:
        try:
            return json.loads(c)
        except json.JSONDecodeError:
            continue
    return {}
