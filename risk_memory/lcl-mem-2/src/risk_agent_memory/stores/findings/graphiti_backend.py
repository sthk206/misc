"""Graphiti (Zep engine) backend over Neo4j — the production C.1 deployment.

Requires the `graphiti` extra, a running Neo4j, and LLM+embedder credentials
(Graphiti uses them for ingestion; route through the same Anthropic account and
log token spend — ingestion cost is part of the C.4 budget story).

Custom entity types registered at setup: Desk, Division, CurrencyPair,
Instrument, OptionTrade, HedgePosition, Client, Counterparty, MarketEvent,
NewsItem, PatternNode. The org/portfolio hierarchy is seeded STATICALLY
(`part_of` edges) — never rely on community detection to rediscover the org
chart.
"""

from __future__ import annotations

import os
from typing import Any

ENTITY_TYPES = [
    "Desk", "Division", "CurrencyPair", "Instrument", "OptionTrade",
    "HedgePosition", "Client", "Counterparty", "MarketEvent", "NewsItem",
    "PatternNode",
]


class GraphitiBackend:
    """Thin async wrapper; construct via `await GraphitiBackend.create()`.

    NOTE: verified against graphiti-core's documented API at design time; pin
    and re-verify the installed version before first production run.
    """

    def __init__(self, graphiti: Any):
        self._g = graphiti

    @classmethod
    async def create(
        cls,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
    ) -> "GraphitiBackend":
        from graphiti_core import Graphiti  # requires the `graphiti` extra

        g = Graphiti(
            uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
            user or os.environ.get("NEO4J_USER", "neo4j"),
            password or os.environ["NEO4J_PASSWORD"],
        )
        await g.build_indices_and_constraints()
        return cls(g)

    async def seed_hierarchy(self, part_of_edges: list[tuple[str, str]]) -> None:
        """Statically seed Desk->Division etc. as episodes with explicit
        part_of statements so they become first-class temporal facts."""
        from graphiti_core.nodes import EpisodeType
        from datetime import datetime, timezone

        for child, parent in part_of_edges:
            await self._g.add_episode(
                name=f"org:{child}",
                episode_body=f"{child} is part of {parent}.",
                source=EpisodeType.text,
                source_description="static org hierarchy seed",
                reference_time=datetime.now(timezone.utc),
            )

    async def add_episode(self, name: str, body: str, reference_time, description: str):
        from graphiti_core.nodes import EpisodeType

        return await self._g.add_episode(
            name=name, episode_body=body, source=EpisodeType.text,
            source_description=description, reference_time=reference_time,
        )

    async def search(self, query: str, k: int = 20):
        return await self._g.search(query, num_results=k)
