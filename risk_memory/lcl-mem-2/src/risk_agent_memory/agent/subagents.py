"""Subagent definitions (spec §2): reflector (A.3), insight_writer (C.4),
investigator (C.5 escalation). Run as SDK subagents to keep their token usage
out of the main context."""

from __future__ import annotations

from claude_agent_sdk import AgentDefinition

from risk_agent_memory.config import CONFIG
from risk_agent_memory.stores.ace.reflector import REFLECTOR_PROMPT
from risk_agent_memory.stores.findings.writer import INSIGHT_WRITER_PROMPT

INVESTIGATOR_PROMPT = """\
You are the investigator subagent, spawned when composed retrieval could not
explain the situation (e.g. DoD attribution coverage < 90%, or the
answerability check failed). Run the iterative loop:
1. Treat the intermediate inferences you were given as new retrieval seeds.
2. Re-scope: query atom tools and the findings store for the entities/periods
   those inferences implicate.
3. Re-compose an explanation with epistemic tags per claim
   (observed | inferred | world_knowledge).
Depth cap: {depth_cap} iterations. If still unexplained, return a partial
answer with an explicit "unattributed residual" statement — never paper over
the gap.
"""


def build_subagents() -> dict[str, AgentDefinition]:
    return {
        "reflector": AgentDefinition(
            description="Post-session playbook reflection: emits ACE deltas only",
            prompt=REFLECTOR_PROMPT,
            tools=["Read", "Grep"],
            model=CONFIG.models.subagent_model,
        ),
        "insight_writer": AgentDefinition(
            description="Writes validated insight objects after investigations",
            prompt=INSIGHT_WRITER_PROMPT,
            tools=["Read", "Grep"],
            model=CONFIG.models.subagent_model,
        ),
        "investigator": AgentDefinition(
            description="Escalation loop when composed retrieval cannot explain "
                        "a move (coverage below threshold)",
            prompt=INVESTIGATOR_PROMPT.format(
                depth_cap=CONFIG.thresholds.escalation_depth_cap
            ),
            tools=["Read", "Grep",
                   "mcp__atom__get_risk_snapshot", "mcp__atom__get_positions",
                   "mcp__atom__get_trade_events", "mcp__atom__get_news"],
            model=CONFIG.models.agent_model,
        ),
    }
