"""A.3 write path: the reflector subagent's contract and delta application.

The reflector runs over the session transcript at Stop time and emits DELTAS
ONLY (never a monolithic playbook rewrite). ADD/MERGE/RETIRE queue for human
review; counters apply automatically; PREF_CANDIDATE routes to the S2
confirmation queue (the reflector must not write preference content directly —
single-ownership rule)."""

from __future__ import annotations

import json
from typing import Any

from risk_agent_memory.stores.ace.models import AceStore

REFLECTOR_PROMPT = """\
You are the playbook reflector for a risk-intelligence agent. You are given the
transcript of one completed session. Review how the agent performed and emit
playbook DELTAS as a JSON list. Never rewrite the playbook wholesale.

Allowed delta kinds:
- {"kind": "ADD", "text": "<imperative directive, <= 60 tokens, no case details>",
   "scope": "global" | "manager:<id>" | "mode:<morning|predef|adhoc>",
   "evidence_span": "<short transcript quote>"}
- {"kind": "INCR", "entry_id": <int>, "direction": "helpful" | "harmful",
   "evidence_span": "<short transcript quote>"}
- {"kind": "MERGE", "entry_ids": [<int>, ...], "text": "<merged directive>"}
- {"kind": "RETIRE", "entry_id": <int>, "reason": "<why>"}
- {"kind": "PREF_CANDIDATE", "manager_id": "<id>", "key": "<registry key>",
   "value": <json>, "evidence_span": "<quote>"}

Rules:
- Directives are procedures ("Always check X when Y"), never case conclusions;
  conclusions belong to the findings store.
- Preference-like observations ("manager keeps asking for EUR/USD first") must be
  PREF_CANDIDATE, never ADD.
- INCR helpful when a fired playbook entry demonstrably helped; harmful when it
  misled. Cite the entry id shown in the injected playbook block ([id|scope]).
- Output ONLY the JSON list, nothing else. Output [] if no changes are warranted.
"""


class DeltaParseError(ValueError):
    pass


def parse_deltas(text: str) -> list[dict[str, Any]]:
    s = text.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        if s.startswith("json"):
            s = s[4:].strip()
    try:
        deltas = json.loads(s)
    except json.JSONDecodeError as e:
        i, j = s.find("["), s.rfind("]")
        if i == -1 or j <= i:
            raise DeltaParseError(str(e)) from e
        deltas = json.loads(s[i : j + 1])
    if not isinstance(deltas, list):
        raise DeltaParseError("reflector output must be a JSON list")
    return deltas


def apply_reflection(
    ace: AceStore,
    deltas: list[dict[str, Any]],
    session_ref: str,
    prefs_store=None,
) -> dict[str, int]:
    """Route parsed deltas: ACE deltas to the ACE queue/counters, preference
    candidates to S2. Returns per-kind counts for logging."""
    counts: dict[str, int] = {}
    for d in deltas:
        kind = str(d.get("kind", "")).upper()
        counts[kind] = counts.get(kind, 0) + 1
        if kind == "PREF_CANDIDATE":
            if prefs_store is not None:
                prefs_store.propose_candidate(
                    manager_id=d["manager_id"], key=d["key"], value=d["value"],
                    evidence=d.get("evidence_span"),
                )
            continue
        payload = {k: v for k, v in d.items() if k != "kind"}
        ace.submit_delta(kind, payload, session_ref=session_ref)
    return counts
