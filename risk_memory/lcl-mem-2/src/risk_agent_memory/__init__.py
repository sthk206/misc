"""Memory system for a risk intelligence agent (implementation_v1.md).

Three stores, one lineage chain, plus live tools:
  atom (trading engine)  -> live MCP tools; NEVER memorized; source of truth for numbers.
  S1 ACE playbook store  -> procedural memory, compiled into context every session.
  S2 Preference store    -> per-manager structured profile, injected at session start.
  S3 Findings store      -> temporal KG (facts) + insight DAG + pattern registry.

Lineage (edges point DOWN to evidence; invalidation propagates UP):
  ACE rule -> pattern node -> insight nodes (DAG) -> {facts, atom snapshots, insights}
"""

__version__ = "0.1.0"
