"""SDK hooks (verified against claude-agent-sdk 0.2.x: Stop / PreCompact /
PostToolUse exist; there is no SessionEnd — the spec's fallback to Stop applies).

- PostToolUse: log every atom snapshot id / fact UUID the session touched (the
  insight writer later selects the load-bearing subset — C.4 parent capture).
- PreCompact: flush provenance-relevant tool results to the episode log before
  the SDK compacts them away.
- Stop: enqueue the session for reflection (A.3) + insight write-back (C.4);
  the harness drains the queue after the session, keeping subagent token usage
  out of the main context.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from claude_agent_sdk import HookMatcher

_SNAPSHOT_RE = re.compile(r'"snapshot_id":\s*"([^"]+)"')


@dataclass
class SessionProvenance:
    """Mutable per-session context shared with hooks."""

    session_ref: str
    snapshot_ids: list[str] = field(default_factory=list)
    fact_uuids: list[str] = field(default_factory=list)
    episode_log: list[dict] = field(default_factory=list)
    reflection_queue: list[str] = field(default_factory=list)

    def episode_path(self, episodes_dir: Path) -> Path:
        episodes_dir.mkdir(parents=True, exist_ok=True)
        return episodes_dir / f"{self.session_ref}.jsonl"


def extract_snapshot_ids(tool_output: str) -> list[str]:
    return _SNAPSHOT_RE.findall(tool_output or "")


def _tool_result_text(hook_input: dict) -> str:
    resp = hook_input.get("tool_response")
    if resp is None:
        return ""
    if isinstance(resp, str):
        return resp
    try:
        return json.dumps(resp, default=str)
    except TypeError:
        return str(resp)


def build_hooks(prov: SessionProvenance, episodes_dir: Path) -> dict:
    async def post_tool_use(hook_input, tool_use_id, context):
        text = _tool_result_text(hook_input)
        for sid in extract_snapshot_ids(text):
            if sid not in prov.snapshot_ids:
                prov.snapshot_ids.append(sid)
        prov.episode_log.append({
            "t": time.time(),
            "tool": hook_input.get("tool_name"),
            "input": hook_input.get("tool_input"),
            "output_excerpt": text[:2000],
        })
        return {}

    async def pre_compact(hook_input, tool_use_id, context):
        # flush provenance-relevant results before compaction discards them
        path = prov.episode_path(episodes_dir)
        with open(path, "a") as f:
            for row in prov.episode_log:
                f.write(json.dumps(row, default=str) + "\n")
        prov.episode_log.clear()
        return {}

    async def on_stop(hook_input, tool_use_id, context):
        path = prov.episode_path(episodes_dir)
        with open(path, "a") as f:
            for row in prov.episode_log:
                f.write(json.dumps(row, default=str) + "\n")
        prov.episode_log.clear()
        prov.reflection_queue.append(prov.session_ref)
        return {}

    return {
        "PostToolUse": [HookMatcher(hooks=[post_tool_use])],
        "PreCompact": [HookMatcher(hooks=[pre_compact])],
        "Stop": [HookMatcher(hooks=[on_stop])],
    }
