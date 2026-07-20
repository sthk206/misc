"""ClaudeAgentOptions builders: treatment (all three stores mounted) vs
baseline (stock SDK agent). The two differ ONLY in mounted stores/hooks —
identical model, mock atom, and skills — so every eval isolates the memory
system's contribution (spec 2.1/6.1)."""

from __future__ import annotations

from pathlib import Path

from claude_agent_sdk import ClaudeAgentOptions

from risk_agent_memory.agent.hooks import SessionProvenance, build_hooks
from risk_agent_memory.agent.subagents import build_subagents
from risk_agent_memory.config import CONFIG
from risk_agent_memory.mock_atom.server import MockAtom
from risk_agent_memory.stores.ace.compiler import compile_playbook
from risk_agent_memory.stores.ace.models import AceStore
from risk_agent_memory.stores.prefs.models import (
    PrefsStore,
    render_candidate_prompts,
    render_profile_block,
)
from risk_agent_memory.stores.prefs.tools import create_prefs_server

BASE_SYSTEM_PROMPT = """\
You are a risk intelligence assistant for FX portfolio managers. Numbers come
ONLY from the atom tools (get_risk_snapshot, get_positions, get_trade_events,
get_news, get_reference_data) — never from memory; cite snapshot_id for every
figure. Follow the injected playbook directives and the manager's confirmed
preferences exactly. When a preference-shaped request appears, use prefs_set;
never assume unstated preferences.
"""


def build_treatment_options(
    manager: str,
    mode: str,
    atom: MockAtom,
    ace: AceStore,
    prefs: PrefsStore,
    prov: SessionProvenance,
    episodes_dir: Path | None = None,
    max_turns: int = 20,
) -> ClaudeAgentOptions:
    playbook = compile_playbook(ace, manager, mode)
    profile_block = render_profile_block(prefs.profile(manager), prefs.registry)
    candidates_block = render_candidate_prompts(prefs, manager)
    system = "\n\n".join(
        b for b in (BASE_SYSTEM_PROMPT, playbook.text, profile_block, candidates_block) if b
    )
    return ClaudeAgentOptions(
        system_prompt=system,
        model=CONFIG.models.agent_model,
        mcp_servers={
            "atom": atom.server(),
            "prefs": create_prefs_server(prefs, manager),
        },
        allowed_tools=[
            "mcp__atom__get_risk_snapshot", "mcp__atom__get_positions",
            "mcp__atom__get_trade_events", "mcp__atom__get_news",
            "mcp__atom__get_reference_data",
            "mcp__prefs__prefs_set", "mcp__prefs__prefs_list",
            "mcp__prefs__prefs_delete", "mcp__prefs__prefs_confirm",
        ],
        hooks=build_hooks(prov, episodes_dir or CONFIG.paths.episodes_dir),
        agents=build_subagents(),
        max_turns=max_turns,
        permission_mode="bypassPermissions",
        setting_sources=[],           # hermetic: no user/project settings bleed
    )


def build_baseline_options(
    atom: MockAtom,
    transcripts_dir: Path,
    max_turns: int = 20,
) -> ClaudeAgentOptions:
    """Stock SDK agent: atom tools, project context, native compaction,
    per-session history only. Prior sessions exist ONLY as transcript files the
    agent may grep — the honest 'SDK out of the box' memory story."""
    system = (
        BASE_SYSTEM_PROMPT
        + f"\nTranscripts of your prior sessions are files in {transcripts_dir}; "
          "you may search them with Grep/Read if you need history."
    )
    return ClaudeAgentOptions(
        system_prompt=system,
        model=CONFIG.models.agent_model,
        mcp_servers={"atom": atom.server()},
        allowed_tools=[
            "mcp__atom__get_risk_snapshot", "mcp__atom__get_positions",
            "mcp__atom__get_trade_events", "mcp__atom__get_news",
            "mcp__atom__get_reference_data", "Read", "Grep",
        ],
        cwd=str(transcripts_dir),
        max_turns=max_turns,
        permission_mode="bypassPermissions",
        setting_sources=[],
    )
