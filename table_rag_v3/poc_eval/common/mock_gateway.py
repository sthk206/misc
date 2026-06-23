"""
Deterministic offline stand-in for the gateway, for PLUMBING validation only.

`install()` monkeypatches llm_gateway.embed / chat / chat_text so the full pipeline
(ingestion -> retrieval -> agent loop -> scoring -> report) can be exercised end-to-end
without any API key or network. It does NOT produce meaningful answers -- embeddings are
bag-of-words hashes and chat replies are canned -- so any metrics from --mock runs are
plumbing artifacts, not findings. Run against the real gateway for real numbers.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from poc_eval.common import llm_gateway

_DIM = 64


def _mock_embed(texts: list[str], model: str = "") -> list[list[float]]:
    out = []
    for t in texts:
        vec = [0.0] * _DIM
        for tok in re.findall(r"[a-zA-Z0-9]+", t.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            vec[h % _DIM] += 1.0
        out.append(vec)
    return out


def _first_table_name(text: str) -> str | None:
    m = re.search(r'TABLE "([^"]+)"', text)
    return m.group(1) if m else None


def _mock_chat_text(messages: list[dict[str, Any]], **kwargs) -> str:
    blob = " ".join(m.get("content", "") or "" for m in messages)
    # Repo NL2SQL (offline NL2SQL_SYSTEM_PROMPT)
    if "expert in SQL" in blob:
        m = re.search(r'"table_name":\s*"([^"]+)"', blob)
        name = m.group(1) if m else "t_unknown"
        return f'```sql\nSELECT row_label FROM "{name}" LIMIT 3\n```'
    # Repo TableRAG controller (SYSTEM_EXPLORE_PROMPT) -> terminate with an answer
    if "table-related question answering task" in blob or "solve_subquery" in blob:
        return "<Answer>: MOCK ANSWER"
    if "impartial grader" in blob:        # eval judge
        return "0"
    if "classify the failure" in blob:    # failure taxonomy
        return "reasoning failure"
    if "financial analyst answering questions" in blob:   # baseline RAG
        return '{"answer":"MOCK ANSWER","value":null,"pages":[]}'
    return '{"answer":"MOCK","value":null,"pages":[]}'


class _Fn:
    def __init__(self, sq): self.arguments = json.dumps({"subquery": sq})


class _Call:
    def __init__(self, sq):
        self.id = "call_mock"
        self.function = _Fn(sq)


class _Msg:
    def __init__(self, content, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


def _mock_chat(messages, tools=None, **kwargs):
    """Object-returning mock for the repo's get_chat_result (controller + COMBINE)."""
    blob = " ".join(m.get("content", "") or "" for m in messages if isinstance(m, dict))
    if tools and "table-related question answering task" in blob:   # the decompose controller
        # 1st turn: issue a sub-query (exercises retrieval + NL->SQL); then finalize.
        if not any(isinstance(m, dict) and m.get("role") == "tool" for m in messages):
            return _Msg("", [_Call("total interest rate contracts and total derivative notional 2025")])
        return _Msg("<Answer>: MOCK ANSWER", None)
    return _Msg(_mock_chat_text(messages, **kwargs), None)


def install() -> None:
    llm_gateway.embed = _mock_embed          # type: ignore[assignment]
    llm_gateway.chat_text = _mock_chat_text  # type: ignore[assignment]
    llm_gateway.chat = _mock_chat            # type: ignore[assignment]
