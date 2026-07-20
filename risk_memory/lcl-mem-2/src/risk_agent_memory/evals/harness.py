"""6.1 eval harness — build once, reuse for every suite.

A scenario = fixture stores + mock-atom world + a session script (one or many
sessions) + scoring fns. Treatment and baseline run through identical
ClaudeSDKClient configs differing only in mounted stores. Metrics logged per
run: task score, tokens in/out, tool calls, latency, full transcript for
judge-based scoring.

Live runs require Claude Code auth (the SDK spawns the CLI); everything above
the `run_scenario` line is offline-testable.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from risk_agent_memory.config import CONFIG
from risk_agent_memory.embedding import get_embedder
from risk_agent_memory.mock_atom.fixtures import build_world
from risk_agent_memory.mock_atom.server import MockAtom
from risk_agent_memory.stores.ace.models import AceStore
from risk_agent_memory.stores.findings.backend import InMemoryFactBackend
from risk_agent_memory.stores.findings.dag import FindingsDag
from risk_agent_memory.stores.prefs.models import PrefsStore
from risk_agent_memory.stores.prefs.registry import PrefRegistry


@dataclass
class SessionScript:
    prompt: str
    day: int = 0
    manager: str = "mgr_a"
    mode: str = "morning"                  # morning | predef | adhoc
    feedback: str | None = None            # follow-up feedback message (StreamBench protocol)


@dataclass
class ScenarioContext:
    workdir: Path
    atom: MockAtom
    ace: AceStore
    prefs: PrefsStore
    dag: FindingsDag
    backend: InMemoryFactBackend
    transcripts: list[str] = field(default_factory=list)
    session_metrics: list[dict] = field(default_factory=list)


@dataclass
class Scenario:
    name: str
    sessions: list[SessionScript]
    setup: Callable[[ScenarioContext], None] | None = None
    score: Callable[[ScenarioContext], dict[str, Any]] | None = None
    world_seed: int = 0
    n_days: int = 30
    embedder_name: str = "intfloat/e5-base-v2"


def make_context(scenario: Scenario, workdir: str | Path) -> ScenarioContext:
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    embedder = get_embedder(scenario.embedder_name)
    ctx = ScenarioContext(
        workdir=workdir,
        atom=MockAtom(build_world(scenario.world_seed, scenario.n_days)),
        ace=AceStore(workdir / "ace.sqlite", embedder),
        prefs=PrefsStore(
            workdir / "prefs.sqlite", PrefRegistry.load(CONFIG.paths.prefs_registry)
        ),
        dag=FindingsDag(workdir / "findings.sqlite", embedder),
        backend=InMemoryFactBackend(),
    )
    if scenario.setup:
        scenario.setup(ctx)
    return ctx


# --------------------------------------------------------------- live runner

async def _run_one_session(options, prompt: str) -> tuple[str, dict]:
    """One SDK session; returns (assistant transcript text, metrics)."""
    from claude_agent_sdk import (
        AssistantMessage, ClaudeSDKClient, ResultMessage, TextBlock, ToolUseBlock,
    )

    t0 = time.time()
    texts: list[str] = []
    tool_calls = 0
    usage: dict = {}
    cost = None
    async with ClaudeSDKClient(options=options) as client:
        await client.query(prompt)
        async for msg in client.receive_response():
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        texts.append(block.text)
                    elif isinstance(block, ToolUseBlock):
                        tool_calls += 1
            elif isinstance(msg, ResultMessage):
                usage = msg.usage or {}
                cost = msg.total_cost_usd
    return "\n".join(texts), {
        "tool_calls": tool_calls,
        "usage": usage,
        "cost_usd": cost,
        "latency_s": round(time.time() - t0, 2),
    }


async def run_reflection(transcript: str, session_ref: str, ctx: ScenarioContext) -> dict:
    """A.3: reflector pass over the transcript, deltas routed to ACE/S2."""
    from claude_agent_sdk import ClaudeAgentOptions, query, AssistantMessage, TextBlock

    from risk_agent_memory.stores.ace.reflector import (
        REFLECTOR_PROMPT, apply_reflection, parse_deltas,
    )

    playbook_listing = "\n".join(
        f"[{e.id}|{e.scope}] {e.text}" for e in ctx.ace.entries("active")
    )
    out: list[str] = []
    async for msg in query(
        prompt=f"Current playbook:\n{playbook_listing}\n\nSession transcript:\n{transcript}",
        options=ClaudeAgentOptions(
            system_prompt=REFLECTOR_PROMPT,
            model=CONFIG.models.subagent_model,
            max_turns=1,
            setting_sources=[],
        ),
    ):
        if isinstance(msg, AssistantMessage):
            out.extend(b.text for b in msg.content if isinstance(b, TextBlock))
    deltas = parse_deltas("\n".join(out))
    return apply_reflection(ctx.ace, deltas, session_ref, prefs_store=ctx.prefs)


async def run_scenario(
    scenario: Scenario,
    workdir: str | Path,
    treatment: bool = True,
    reflect: bool = True,
) -> ScenarioContext:
    """Run all sessions of a scenario live. Baseline runs write transcript
    files (its only memory); treatment runs reflection after each session."""
    from risk_agent_memory.agent.hooks import SessionProvenance
    from risk_agent_memory.agent.options import (
        build_baseline_options, build_treatment_options,
    )

    ctx = make_context(scenario, workdir)
    transcripts_dir = ctx.workdir / "transcripts"
    transcripts_dir.mkdir(exist_ok=True)

    for si, s in enumerate(scenario.sessions):
        ctx.atom.current_day = s.day
        session_ref = f"{scenario.name}:s{si}"
        if treatment:
            prov = SessionProvenance(session_ref=session_ref)
            options = build_treatment_options(
                s.manager, s.mode, ctx.atom, ctx.ace, ctx.prefs, prov,
                episodes_dir=ctx.workdir / "episodes",
            )
        else:
            options = build_baseline_options(ctx.atom, transcripts_dir)
        text, metrics = await _run_one_session(options, s.prompt)
        if s.feedback:
            fb_text, fb_metrics = await _run_one_session(options,
                f"(continued session)\nPrior answer:\n{text}\n\nManager feedback: {s.feedback}")
            text = text + "\n---feedback follow-up---\n" + fb_text
            metrics["tool_calls"] += fb_metrics["tool_calls"]
        ctx.transcripts.append(text)
        metrics.update({"session": session_ref, "treatment": treatment, "day": s.day})
        ctx.session_metrics.append(metrics)
        (transcripts_dir / f"session_{si:03d}.md").write_text(
            f"# {session_ref} (day {s.day}, {s.manager}, {s.mode})\n\n{text}\n"
        )
        if treatment and reflect:
            await run_reflection(text, session_ref, ctx)
    return ctx


def score_and_write(
    scenario: Scenario, ctx: ScenarioContext, results_dir: str | Path = "results"
) -> dict:
    metrics = scenario.score(ctx) if scenario.score else {}
    out = {
        "scenario": scenario.name,
        "metrics": metrics,
        "sessions": ctx.session_metrics,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"{scenario.name}_{int(time.time())}.json").write_text(
        json.dumps(out, indent=2, default=str)
    )
    return out
