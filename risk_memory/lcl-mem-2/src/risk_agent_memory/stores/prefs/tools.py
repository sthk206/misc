"""B.2 MCP tools for the preference store, served in-process via
create_sdk_mcp_server. `prefs_set` writes confirmed+explicit directly (the
manager stated it); list/delete are the edit/audit surface."""

from __future__ import annotations

import json

from claude_agent_sdk import create_sdk_mcp_server, tool

from risk_agent_memory.stores.prefs.models import PrefsStore
from risk_agent_memory.stores.prefs.registry import InvalidPrefValue, UnknownPrefKey


def _text(payload: str) -> dict:
    return {"content": [{"type": "text", "text": payload}]}


def create_prefs_server(store: PrefsStore, manager_id: str):
    """Build the in-process MCP server bound to one manager's session."""

    @tool(
        "prefs_set",
        "Set a manager preference when the manager explicitly states one "
        "(e.g. 'always show EUR/USD first'). Acknowledge in your reply.",
        {"key": str, "value": str},
    )
    async def prefs_set(args):
        try:
            value = json.loads(args["value"])
        except (json.JSONDecodeError, TypeError):
            value = args["value"]
        try:
            store.set(manager_id, args["key"], value, source="explicit")
        except (UnknownPrefKey, InvalidPrefValue) as e:
            return _text(f"REJECTED: {e}")
        return _text(f"saved: {args['key']} = {json.dumps(value)} (confirmed, explicit)")

    @tool("prefs_list", "List this manager's stored preferences (audit view).", {})
    async def prefs_list(args):
        rows = store.all_rows(manager_id)
        if not rows:
            return _text("no preferences stored")
        return _text(
            "\n".join(
                f"{r.key} = {json.dumps(r.value)} [{r.status}, {r.source}]" for r in rows
            )
        )

    @tool("prefs_delete", "Delete a stored preference (revocation).", {"key": str})
    async def prefs_delete(args):
        store.delete(manager_id, args["key"])
        return _text(f"deleted {args['key']}")

    @tool(
        "prefs_confirm",
        "Confirm a suggested (candidate) preference after the manager says yes.",
        {"key": str},
    )
    async def prefs_confirm(args):
        try:
            store.confirm(manager_id, args["key"])
        except KeyError as e:
            return _text(f"REJECTED: {e}")
        return _text(f"confirmed {args['key']}")

    return create_sdk_mcp_server(
        "prefs", tools=[prefs_set, prefs_list, prefs_delete, prefs_confirm]
    )
