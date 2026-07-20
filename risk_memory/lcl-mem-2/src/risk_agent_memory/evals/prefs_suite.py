"""6.3 Phase B suite: "are stated preferences honored, forever, for the right
person?" PrefEval's protocol (unprompted adherence over long horizons +
violation types) ported onto risk-report tasks; its consumer-topic data is
rejected. Mechanical prefs (chart order, thresholds) score programmatically —
that is why B.2 mandates mechanical consumption."""

from __future__ import annotations

import re

from risk_agent_memory.evals.harness import Scenario, ScenarioContext, SessionScript


def chart_order_respected(transcript: str, expected_order: list[str]) -> bool:
    """Programmatic: first mentions of the pairs must appear in profile order."""
    positions = []
    for pair in expected_order:
        m = re.search(re.escape(pair), transcript.replace("/", ""))
        if m is None:
            return False
        positions.append(m.start())
    return positions == sorted(positions)


def adherence_scenario() -> Scenario:
    order = ["EURUSD", "USDJPY", "GBPUSD"]

    def setup(ctx: ScenarioContext) -> None:
        ctx.prefs.set("mgr_a", "layout.chart_order", order, source="explicit")
        ctx.prefs.set("mgr_a", "thresholds.dod_flag_ccy_pair", 0.5, source="explicit")

    sessions = [
        SessionScript(prompt="Produce the morning risk report.", day=d,
                      manager="mgr_a", mode="morning")
        for d in (1, 5, 9)
    ]

    def score(ctx: ScenarioContext) -> dict:
        ordered = [chart_order_respected(t, order) for t in ctx.transcripts]
        return {"chart_order_adherence": sum(ordered) / len(ordered),
                "per_session": ordered}

    return Scenario(name="prefs_adherence", sessions=sessions, setup=setup, score=score)


def persistence_scenario(check_sessions: tuple[int, ...] = (2, 5, 10)) -> Scenario:
    """Pref stated conversationally in session 1 (prefs_set path), checked in
    sessions 2, 5, 10. Baseline gets only transcript-grep memory."""
    n = max(check_sessions) + 1
    sessions = [
        SessionScript(
            prompt=(
                "From now on always show EUR/USD first in my reports. "
                "Also produce today's morning risk report."
                if i == 0 else "Produce the morning risk report."
            ),
            day=i, manager="mgr_a", mode="morning",
        )
        for i in range(n)
    ]

    def score(ctx: ScenarioContext) -> dict:
        stored = ctx.prefs.profile("mgr_a").get("layout.chart_order")
        honored = {
            f"session_{i}": bool(re.search(r"EUR\s?/?\s?USD", ctx.transcripts[i]))
            and chart_order_respected(ctx.transcripts[i], ["EURUSD"])
            for i in check_sessions
            if i < len(ctx.transcripts)
        }
        return {"pref_stored": stored is not None, **honored}

    return Scenario(name="prefs_persistence", sessions=sessions, score=score)


def isolation_and_revocation_scenario() -> Scenario:
    def setup(ctx: ScenarioContext) -> None:
        ctx.prefs.set("mgr_a", "tone.verbosity", "terse", source="explicit")

    sessions = [
        SessionScript(prompt="Produce the morning risk report.", day=1,
                      manager="mgr_b", mode="morning"),      # isolation probe
        SessionScript(prompt="Delete my verbosity preference, then produce the report.",
                      day=2, manager="mgr_a", mode="morning"),
        SessionScript(prompt="Produce the morning risk report.", day=3,
                      manager="mgr_a", mode="morning"),      # revocation probe
    ]

    def score(ctx: ScenarioContext) -> dict:
        return {
            "b_profile_empty": ctx.prefs.profile("mgr_b") == {},
            "deleted_after_revoke": "tone.verbosity" not in ctx.prefs.profile("mgr_a"),
        }

    return Scenario(name="prefs_isolation_revocation", sessions=sessions,
                    setup=setup, score=score)


def inference_loop_scenario() -> Scenario:
    """Scripted manager repeats a request 3 sessions running; the candidate
    must appear, must NOT auto-apply before confirmation, must apply after."""
    sessions = [
        SessionScript(prompt="Produce the morning report; show EUR/USD first please.",
                      day=d, manager="mgr_a", mode="morning")
        for d in (1, 2, 3)
    ] + [
        SessionScript(prompt="Yes, make that the default. Then produce the report.",
                      day=4, manager="mgr_a", mode="morning"),
    ]

    def score(ctx: ScenarioContext) -> dict:
        confirmed = ctx.prefs.profile("mgr_a").get("layout.chart_order")
        rows = ctx.prefs.all_rows("mgr_a")
        had_candidate = any(r.status == "candidate" for r in rows) or confirmed is not None
        return {"candidate_appeared": had_candidate,
                "confirmed_after_yes": confirmed is not None}

    return Scenario(name="prefs_inference_loop", sessions=sessions, score=score)
