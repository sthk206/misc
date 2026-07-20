"""CLI: review queues, nightly jobs, and eval entry points.

Usage:
  ram playbook review | list | notifications
  ram prefs list <manager> | delete <manager> <key>
  ram findings patterns | promote
  ram eval findings            # offline suite (no LLM)
  ram eval ace|prefs --live    # live SDK runs (needs Claude auth)
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer
from rich.console import Console

from risk_agent_memory.stores.ace.review_cli import playbook_app

app = typer.Typer(no_args_is_help=True)
app.add_typer(playbook_app, name="playbook")
console = Console()

prefs_app = typer.Typer(no_args_is_help=True, help="Preference store audit/edit")
app.add_typer(prefs_app, name="prefs")
findings_app = typer.Typer(no_args_is_help=True, help="Findings store jobs")
app.add_typer(findings_app, name="findings")
eval_app = typer.Typer(no_args_is_help=True, help="Eval suites")
app.add_typer(eval_app, name="eval")


def _stores():
    from risk_agent_memory.config import CONFIG
    from risk_agent_memory.embedding import get_embedder
    from risk_agent_memory.stores.ace.models import AceStore
    from risk_agent_memory.stores.findings.dag import FindingsDag
    from risk_agent_memory.stores.prefs.models import PrefsStore
    from risk_agent_memory.stores.prefs.registry import PrefRegistry

    emb = get_embedder(CONFIG.models.embedder)
    return (
        AceStore(CONFIG.paths.ace_db, emb),
        PrefsStore(CONFIG.paths.prefs_db, PrefRegistry.load(CONFIG.paths.prefs_registry)),
        FindingsDag(CONFIG.paths.findings_db, emb),
    )


@prefs_app.command("list")
def prefs_list(manager: str = typer.Argument(None)):
    _, prefs, _ = _stores()
    for r in prefs.all_rows(manager):
        console.print(
            f"{r.manager_id}  {r.key} = {json.dumps(r.value)} "
            f"[{r.status}, {r.source}]"
        )


@prefs_app.command("delete")
def prefs_delete(manager: str, key: str):
    _, prefs, _ = _stores()
    prefs.delete(manager, key)
    console.print(f"deleted {manager}/{key}")


@findings_app.command("patterns")
def findings_patterns():
    _, _, dag = _stores()
    for p in dag.patterns():
        console.print(f"[{p.id}] {p.name} — live {p.live_instances}, "
                      f"instances {p.instance_insight_ids}")
    for row in dag.pattern_review_queue():
        console.print(f"[yellow]review[/yellow] {row['name']}: {row['description']}")


@findings_app.command("promote")
def findings_promote():
    """C.7 nightly promotion scan -> ACE candidate drafts."""
    ace, _, dag = _stores()
    drafted = __import__(
        "risk_agent_memory.stores.findings.promotion", fromlist=["scan_and_promote"]
    ).scan_and_promote(dag, ace)
    console.print(f"drafted {len(drafted)} ACE candidates: {drafted}")


@eval_app.command("findings")
def eval_findings(embedder: str = "intfloat/e5-base-v2"):
    """Offline findings suite (retrieval + invalidation + temporal, no LLM)."""
    from risk_agent_memory.evals.findings_suite import run_offline_suite

    metrics = run_offline_suite(embedder_name=embedder)
    console.print_json(data=metrics)
    out = Path("results/findings_offline.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2, default=str))


@eval_app.command("ace")
def eval_ace(
    scenario: str = typer.Option("scope_isolation", help="adherence|learning_stream|scope_isolation"),
    baseline: bool = typer.Option(False, help="run the baseline arm instead of treatment"),
    workdir: Path = Path("results/ace_runs"),
):
    """Live SDK run of a Phase A scenario (needs Claude auth)."""
    from risk_agent_memory.evals import ace_suite
    from risk_agent_memory.evals.harness import run_scenario, score_and_write

    sc = {
        "adherence": ace_suite.adherence_scenario,
        "learning_stream": ace_suite.learning_stream_scenario,
        "scope_isolation": ace_suite.scope_isolation_scenario,
    }[scenario]()
    ctx = asyncio.run(run_scenario(sc, workdir / scenario, treatment=not baseline))
    console.print_json(data=score_and_write(sc, ctx))


@eval_app.command("prefs")
def eval_prefs(
    scenario: str = typer.Option("adherence",
                                 help="adherence|persistence|isolation|inference"),
    baseline: bool = typer.Option(False),
    workdir: Path = Path("results/prefs_runs"),
):
    """Live SDK run of a Phase B scenario (needs Claude auth)."""
    from risk_agent_memory.evals import prefs_suite
    from risk_agent_memory.evals.harness import run_scenario, score_and_write

    sc = {
        "adherence": prefs_suite.adherence_scenario,
        "persistence": prefs_suite.persistence_scenario,
        "isolation": prefs_suite.isolation_and_revocation_scenario,
        "inference": prefs_suite.inference_loop_scenario,
    }[scenario]()
    ctx = asyncio.run(run_scenario(sc, workdir / scenario, treatment=not baseline))
    console.print_json(data=score_and_write(sc, ctx))


if __name__ == "__main__":
    app()
