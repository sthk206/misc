"""A.2 Injection path: deterministic playbook compilation, no retrieval.

All `active` entries matching (manager, mode), ordered by scope specificity
then recency, rendered into a `## Playbook` block appended to the system
prompt. Hard budget: 2,000 tokens. Over budget: lowest-scoring entries
(helpful - harmful, tie-break stale last_fired_at) are dropped and flagged to
the pruning queue. The budget is the forcing function — fix curation, never
raise it to fix eval scores.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from risk_agent_memory.config import CONFIG
from risk_agent_memory.stores.ace.models import (
    AceStore,
    PlaybookEntry,
    approx_tokens,
    scope_specificity,
)

HEADER = "## Playbook\n"


@dataclass
class CompiledPlaybook:
    text: str
    included: list[PlaybookEntry] = field(default_factory=list)
    dropped: list[PlaybookEntry] = field(default_factory=list)
    tokens: int = 0


def _render(entry: PlaybookEntry) -> str:
    return f"- [{entry.id}|{entry.scope}] {entry.text}"


def compile_playbook(
    store: AceStore,
    manager: str,
    mode: str,
    budget: int | None = None,
) -> CompiledPlaybook:
    budget = budget or CONFIG.thresholds.ace_token_budget
    entries = store.active_for(manager, mode)
    # scope specificity desc, then recency desc — deterministic
    entries.sort(key=lambda e: (-scope_specificity(e.scope), -e.created_at, e.id))

    total = approx_tokens(HEADER)
    kept = list(entries)
    line_tokens = {e.id: approx_tokens(_render(e)) for e in kept}
    total += sum(line_tokens.values())

    dropped: list[PlaybookEntry] = []
    while total > budget and kept:
        # drop worst: lowest (helpful - harmful), tie-break stalest last_fired_at
        worst = min(
            kept, key=lambda e: (e.score, e.last_fired_at or 0.0, -e.id)
        )
        kept.remove(worst)
        dropped.append(worst)
        total -= line_tokens[worst.id]
        store.flag_for_pruning(worst.id, "over_budget")

    text = HEADER + "\n".join(_render(e) for e in kept) if kept else ""
    return CompiledPlaybook(text=text, included=kept, dropped=dropped, tokens=total)
