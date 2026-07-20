"""In-process MCP server serving deterministic mock-atom fixtures — this is
what makes every eval runnable inside the SDK rather than as detached
retrieval unit tests (spec 6.1).

The `current_day` is scenario state so multi-session evals can advance time.
"""

from __future__ import annotations

import json

from claude_agent_sdk import create_sdk_mcp_server, tool

from risk_agent_memory.mock_atom.fixtures import DESKS, AtomWorld


def _text(payload) -> dict:
    if not isinstance(payload, str):
        payload = json.dumps(payload, indent=1, default=str)
    return {"content": [{"type": "text", "text": payload}]}


class MockAtom:
    """Holds world + clock; exposes the MCP server and a snapshot-access log
    (consumed by the PostToolUse hook for provenance)."""

    def __init__(self, world: AtomWorld, current_day: int = 0):
        self.world = world
        self.current_day = current_day
        self.accessed_snapshot_ids: list[str] = []

    def server(self):
        atom = self

        @tool(
            "get_risk_snapshot",
            "Current risk snapshot: VaR per desk and day-over-day % move per "
            "currency pair. Returns snapshot_id for citation.",
            {},
        )
        async def get_risk_snapshot(args):
            s = atom.world.snapshot(atom.current_day)
            atom.accessed_snapshot_ids.append(s.snapshot_id)
            return _text({
                "snapshot_id": s.snapshot_id, "hash": s.hash, "day": s.day,
                "desk_var_musd": s.desk_var, "pair_dod_pct": s.pair_dod_pct,
            })

        @tool(
            "get_positions",
            "Open positions, optionally filtered by desk.",
            {"desk": str},
        )
        async def get_positions(args):
            s = atom.world.snapshot(atom.current_day)
            atom.accessed_snapshot_ids.append(s.snapshot_id)
            desk = args.get("desk", "")
            pos = [p for p in s.positions if not desk or p["desk"] == desk]
            return _text({"snapshot_id": s.snapshot_id, "positions": pos})

        @tool(
            "get_trade_events",
            "Trade lifecycle events (open/amend/cancel/expire/correct) for a "
            "day; day=-1 means today.",
            {"day": int},
        )
        async def get_trade_events(args):
            day = args.get("day", -1)
            if day is None or day < 0:
                day = atom.current_day
            evs = atom.world.events_on(day)
            return _text([e.__dict__ for e in evs])

        @tool("get_news", "News items for a day; day=-1 means today.", {"day": int})
        async def get_news(args):
            day = args.get("day", -1)
            if day is None or day < 0:
                day = atom.current_day
            return _text([n for n in atom.world.news if n["day"] == day])

        @tool("get_reference_data", "Org hierarchy: desks and their divisions.", {})
        async def get_reference_data(args):
            return _text({"desks": DESKS})

        return create_sdk_mcp_server(
            "atom",
            tools=[get_risk_snapshot, get_positions, get_trade_events, get_news,
                   get_reference_data],
        )
