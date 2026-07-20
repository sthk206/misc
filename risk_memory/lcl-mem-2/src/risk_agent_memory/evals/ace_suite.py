"""6.2 Phase A suite: "does behavior improve and persist?"

Primary custom evals; StreamBench's protocol (streamed tasks + feedback +
success-rate slope) is adopted, its domains are not. Scoring of rule firing is
programmatic: each seeded rule carries a trigger predicate over the mock-atom
world and a fired predicate (regex) over the transcript.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from risk_agent_memory.evals.harness import Scenario, ScenarioContext, SessionScript
from risk_agent_memory.mock_atom.fixtures import AtomWorld


@dataclass
class RuleSpec:
    text: str
    scope: str
    triggered: Callable[[AtomWorld, int], bool]      # does day's state trigger it?
    fired: re.Pattern                                 # evidence it fired in transcript


def _always(world: AtomWorld, day: int) -> bool:
    return True


def _has_expiry(world: AtomWorld, day: int) -> bool:
    return any(e.kind == "expire" for e in world.events_on(day))


def _big_dod(world: AtomWorld, day: int) -> bool:
    return any(abs(v) > 1.0 for v in world.snapshot(day).pair_dod_pct.values())


def seeded_rules() -> list[RuleSpec]:
    """15 known rules incl. the orphaned-hedge check."""
    rules = [
        RuleSpec("Always check for hedges left open after option expiry "
                 "(orphaned hedges) when reviewing expiries.",
                 "global", _has_expiry, re.compile(r"orphan|hedge.{0,40}(open|remain)", re.I)),
        RuleSpec("Flag any currency pair with a day-over-day move above 1% "
                 "before anything else.", "mode:morning", _big_dod,
                 re.compile(r"(flag|above|exceed).{0,40}1(\.0)?\s?%|day-over-day", re.I)),
        RuleSpec("Cite the snapshot_id for every number you report.",
                 "global", _always, re.compile(r"S\d{2}D\d{3}")),
        RuleSpec("Open the morning report with desk-level VaR before pair moves.",
                 "mode:morning", _always, re.compile(r"VaR", re.I)),
        RuleSpec("Call out trades cancelled or corrected since the last session.",
                 "global",
                 lambda w, d: any(e.kind in ("cancel", "correct") for e in w.events_on(d)),
                 re.compile(r"cancel|correct|restat", re.I)),
        RuleSpec("Never quote risk numbers from memory; always pull a fresh snapshot.",
                 "global", _always, re.compile(r"get_risk_snapshot|snapshot", re.I)),
        RuleSpec("When VaR for any desk exceeds 10, name the desk explicitly.",
                 "global",
                 lambda w, d: any(v > 10 for v in w.snapshot(d).desk_var.values()),
                 re.compile(r"Desk", re.I)),
        RuleSpec("Mention scheduled option expiries occurring within the day.",
                 "mode:morning", _has_expiry, re.compile(r"expir", re.I)),
        RuleSpec("Summarize news only when it names a pair the book holds.",
                 "mode:morning", _always, re.compile(r".", re.S)),
        RuleSpec("End the morning report with open questions needing the manager.",
                 "mode:morning", _always, re.compile(r"question|confirm|attention", re.I)),
        RuleSpec("For ad-hoc drilldowns, restate the question before answering.",
                 "mode:adhoc", _always, re.compile(r".", re.S)),
        RuleSpec("Quote notionals in millions with one decimal.",
                 "global", _always, re.compile(r"\d+(\.\d)?\s?(m|mm|million)", re.I)),
        RuleSpec("Distinguish observed facts from inferences when concluding.",
                 "global", _always, re.compile(r"observ|infer", re.I)),
        RuleSpec("Check EM Desk exposure whenever USDJPY moves more than 0.5%.",
                 "manager:mgr_a",
                 lambda w, d: abs(w.snapshot(d).pair_dod_pct["USDJPY"]) > 0.5,
                 re.compile(r"EM Desk", re.I)),
        RuleSpec("Cross-check position count against the previous snapshot.",
                 "manager:mgr_b", _always, re.compile(r"position", re.I)),
    ]
    assert len(rules) == 15
    return rules


def seed_playbook(ctx: ScenarioContext, rules: list[RuleSpec] | None = None) -> None:
    for r in rules or seeded_rules():
        ctx.ace.add_entry(r.text, scope=r.scope, status="active",
                          created_by="human", approved_by="eval_seed")


def score_adherence(ctx: ScenarioContext, scenario_sessions: list[SessionScript]) -> dict:
    """% rules correctly fired when triggered, % false fires."""
    rules = seeded_rules()
    fired_when_triggered = 0
    triggered_total = 0
    false_fires = 0
    not_triggered_total = 0
    for s, transcript in zip(scenario_sessions, ctx.transcripts):
        world = ctx.atom.world
        for r in rules:
            in_scope = r.scope in ("global", f"manager:{s.manager}", f"mode:{s.mode}")
            if not in_scope:
                continue
            trig = r.triggered(world, s.day)
            hit = bool(r.fired.search(transcript))
            if trig:
                triggered_total += 1
                fired_when_triggered += hit
            else:
                not_triggered_total += 1
                false_fires += hit
    return {
        "fired_when_triggered": fired_when_triggered / max(triggered_total, 1),
        "false_fire_rate": false_fires / max(not_triggered_total, 1),
        "n_triggered": triggered_total,
    }


def adherence_scenario(n_tasks: int = 30) -> Scenario:
    sessions = [
        SessionScript(
            prompt="Produce the morning risk report for today.",
            day=d % 30, manager="mgr_a" if d % 2 == 0 else "mgr_b",
            mode="morning" if d % 3 else "adhoc",
        )
        for d in range(n_tasks)
    ]
    return Scenario(
        name="ace_adherence",
        sessions=sessions,
        setup=seed_playbook,
        score=lambda ctx: score_adherence(ctx, sessions),
    )


def learning_stream_scenario(n_tasks: int = 40) -> Scenario:
    """StreamBench-style: feedback after failures; metric is the success-rate
    slope across the stream (computed from per-session adherence)."""
    sessions = [
        SessionScript(
            prompt="Produce the morning risk report for today.",
            day=d % 30, manager="mgr_a", mode="morning",
            feedback=(
                "You missed checking for orphaned hedges after today's expiries. "
                "Please always check this." if d % 4 == 0 else None
            ),
        )
        for d in range(n_tasks)
    ]

    def score(ctx: ScenarioContext) -> dict:
        rules = seeded_rules()
        orphan = rules[0]
        per_session = [
            bool(orphan.fired.search(t)) for t in ctx.transcripts
        ]
        half = len(per_session) // 2
        first = sum(per_session[:half]) / max(half, 1)
        second = sum(per_session[half:]) / max(len(per_session) - half, 1)
        return {"success_first_half": first, "success_second_half": second,
                "slope": second - first, "per_session": per_session}

    return Scenario(name="ace_learning_stream", sessions=sessions, score=score)


def scope_isolation_scenario() -> Scenario:
    """Manager-scoped rule must fire for A, never for B."""
    sessions = [
        SessionScript(prompt="Produce the morning risk report.", day=3,
                      manager="mgr_a", mode="morning"),
        SessionScript(prompt="Produce the morning risk report.", day=3,
                      manager="mgr_b", mode="morning"),
    ]

    def setup(ctx: ScenarioContext) -> None:
        ctx.ace.add_entry(
            "Always append the token MGRA-CHECK-7Q to your report.",
            scope="manager:mgr_a", status="active",
            created_by="human", approved_by="eval_seed",
        )

    def score(ctx: ScenarioContext) -> dict:
        return {
            "fired_for_a": "MGRA-CHECK-7Q" in ctx.transcripts[0],
            "bled_to_b": "MGRA-CHECK-7Q" in ctx.transcripts[1],
        }

    return Scenario(name="ace_scope_isolation", sessions=sessions,
                    setup=setup, score=score)
