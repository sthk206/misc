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
    system = messages[0]["content"] if messages else ""
    last = messages[-1]["content"] if messages else ""
    if "translate a question into ONE SQLite SELECT" in system:
        name = _first_table_name(last) or "t_unknown"
        return f'{{"table":"{name}","sql":"SELECT row_label, c1 FROM \\"{name}\\" LIMIT 3"}}'
    if "You are TableRAG" in system:
        # Issue one real sub-query (exercises retrieval + NL->SQL), then finalize.
        if not last.startswith("Observation"):
            return '{"action":"query","subquery":"mock sub-query"}'
        return '{"action":"final","answer":"MOCK ANSWER","value":null,"pages":[]}'
    if "financial analyst answering questions" in system:
        return '{"answer":"MOCK ANSWER","value":null,"pages":[]}'
    if "impartial grader" in system:
        return "0"
    if "classify the failure" in system:
        return "reasoning failure"
    return '{"answer":"MOCK","value":null,"pages":[]}'


def _mock_chat(messages, **kwargs):
    class _M:
        content = _mock_chat_text(messages, **kwargs)
        tool_calls = None
    return _M()


def install() -> None:
    llm_gateway.embed = _mock_embed          # type: ignore[assignment]
    llm_gateway.chat_text = _mock_chat_text  # type: ignore[assignment]
    llm_gateway.chat = _mock_chat            # type: ignore[assignment]
