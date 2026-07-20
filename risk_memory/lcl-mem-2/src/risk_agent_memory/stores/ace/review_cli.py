"""A.3.4 human approval CLI: `ram playbook review` lists candidates with
evidence spans; approve / edit / reject. Nothing becomes active without this."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

playbook_app = typer.Typer(no_args_is_help=True, help="Playbook review & inspection")
console = Console()


def _store():
    from risk_agent_memory.config import CONFIG
    from risk_agent_memory.embedding import get_embedder
    from risk_agent_memory.stores.ace.models import AceStore

    return AceStore(CONFIG.paths.ace_db, get_embedder(CONFIG.models.embedder))


@playbook_app.command("review")
def review():
    """Interactively review pending deltas (a)pprove / (e)dit / (r)eject / (s)kip."""
    store = _store()
    pending = store.pending_deltas()
    if not pending:
        console.print("[green]no pending deltas[/green]")
        raise typer.Exit()
    for d in pending:
        console.rule(f"delta {d['id']} — {d['kind']}")
        console.print_json(data=d["payload"])
        if d["session_ref"]:
            console.print(f"[dim]session: {d['session_ref']}[/dim]")
        choice = typer.prompt("approve/edit/reject/skip [a/e/r/s]", default="s")
        if choice.lower().startswith("a"):
            entry_id = store.decide_delta(d["id"], True, by="human")
            console.print(f"[green]approved[/green] -> entry {entry_id}")
        elif choice.lower().startswith("e"):
            payload = dict(d["payload"])
            payload["text"] = typer.prompt("edited text", default=payload.get("text", ""))
            entry_id = store.decide_delta(d["id"], True, by="human", edited_payload=payload)
            console.print(f"[green]approved (edited)[/green] -> entry {entry_id}")
        elif choice.lower().startswith("r"):
            store.decide_delta(d["id"], False, by="human")
            console.print("[red]rejected[/red]")


@playbook_app.command("list")
def list_entries(status: str = typer.Option(None, help="candidate|active|retired")):
    store = _store()
    table = Table("id", "scope", "status", "score", "text", "justification")
    for e in store.entries(status):
        table.add_row(
            str(e.id), e.scope, e.status, str(e.score), e.text,
            e.justification_ptr or "-",
        )
    console.print(table)


@playbook_app.command("notifications")
def notifications():
    """Open notifications (e.g. 'evidence weakened: reconfirm or retire')."""
    store = _store()
    for n in store.open_notifications():
        console.print(f"[yellow]{n['kind']}[/yellow] {n['payload']}")
