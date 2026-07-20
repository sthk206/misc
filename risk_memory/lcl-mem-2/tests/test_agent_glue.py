"""SDK-facing glue: options build against the installed claude-agent-sdk,
hooks capture provenance, mock atom serves deterministic fixtures."""

import asyncio
import json

from risk_agent_memory.agent.hooks import (
    SessionProvenance,
    build_hooks,
    extract_snapshot_ids,
)
from risk_agent_memory.evals.ace_suite import adherence_scenario, seeded_rules
from risk_agent_memory.evals.harness import make_context
from risk_agent_memory.mock_atom.fixtures import build_world
from risk_agent_memory.mock_atom.server import MockAtom


def test_world_is_deterministic():
    w1, w2 = build_world(seed=7, n_days=10), build_world(seed=7, n_days=10)
    assert w1.to_dict() == w2.to_dict()
    assert build_world(seed=8, n_days=10).to_dict() != w1.to_dict()


def test_snapshot_hash_stable():
    w = build_world(seed=1, n_days=3)
    assert w.snapshot(1).hash == build_world(seed=1, n_days=3).snapshot(1).hash


def test_extract_snapshot_ids():
    out = json.dumps({"snapshot_id": "S00D003", "pair_dod_pct": {}})
    assert extract_snapshot_ids(out) == ["S00D003"]
    assert extract_snapshot_ids("no ids here") == []


def test_hooks_capture_and_flush(tmp_path):
    prov = SessionProvenance(session_ref="t:s0")
    hooks = build_hooks(prov, tmp_path)
    assert set(hooks) == {"PostToolUse", "PreCompact", "Stop"}
    post = hooks["PostToolUse"][0].hooks[0]
    stop = hooks["Stop"][0].hooks[0]

    async def run():
        await post(
            {"tool_name": "mcp__atom__get_risk_snapshot",
             "tool_input": {},
             "tool_response": json.dumps({"snapshot_id": "S00D001"})},
            None, None,
        )
        await stop({}, None, None)

    asyncio.run(run())
    assert prov.snapshot_ids == ["S00D001"]
    assert prov.reflection_queue == ["t:s0"]
    logged = (tmp_path / "t:s0.jsonl").read_text().strip().splitlines()
    assert len(logged) == 1
    assert json.loads(logged[0])["tool"] == "mcp__atom__get_risk_snapshot"


def test_options_build_against_installed_sdk(tmp_path):
    from risk_agent_memory.agent.options import (
        build_baseline_options, build_treatment_options,
    )

    scenario = adherence_scenario(n_tasks=1)
    scenario.embedder_name = "hash-32"
    ctx = make_context(scenario, tmp_path / "ctx")
    prov = SessionProvenance(session_ref="t:s0")
    opts = build_treatment_options(
        "mgr_a", "morning", ctx.atom, ctx.ace, ctx.prefs, prov,
        episodes_dir=tmp_path / "episodes",
    )
    # playbook injection landed in the system prompt within budget
    assert "## Playbook" in opts.system_prompt
    assert "orphaned hedges" in opts.system_prompt
    assert set(opts.mcp_servers) == {"atom", "prefs"}
    assert set(opts.agents) == {"reflector", "insight_writer", "investigator"}

    base = build_baseline_options(ctx.atom, tmp_path / "transcripts")
    assert "## Playbook" not in base.system_prompt
    assert set(base.mcp_servers) == {"atom"}


def test_seeded_rules_fit_budget(tmp_path):
    from risk_agent_memory.evals.ace_suite import seed_playbook
    from risk_agent_memory.evals.harness import make_context
    from risk_agent_memory.stores.ace.compiler import compile_playbook

    scenario = adherence_scenario(n_tasks=1)
    scenario.embedder_name = "hash-32"
    ctx = make_context(scenario, tmp_path / "ctx")
    pb = compile_playbook(ctx.ace, "mgr_a", "morning")
    assert not pb.dropped                      # 15 real rules fit in 2k tokens
    assert len(pb.included) >= 13              # all in-scope rules present
    assert len(seeded_rules()) == 15
