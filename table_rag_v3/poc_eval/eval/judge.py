"""
LLM-as-judge scoring + failure taxonomy.

`score_answer` returns 1/0 (correct/incorrect) using an impartial-grader prompt; the
rubric mirrors the repo's EVALUATION_PRONPT (semantic match, numeric accuracy, no
hallucination) but is phrased so mock_gateway routes it ("impartial grader"). It
accepts either the repo's "Rating: [[1]]" format or a bare 0/1.

`classify_failure` buckets an incorrect answer into a coarse category, on a prompt
containing "classify the failure" (the mock contract).
"""
from __future__ import annotations

import re
from typing import Optional

from poc_eval.common import llm_gateway

JUDGE_PROMPT = """You are an impartial grader of answers to questions about a financial 10-K filing.
Compare the assistant's predicted answer to the gold answer. Judge on:
  - Accuracy and hallucinations: numerically and semantically consistent with the gold answer; the numeric value (and units/order of magnitude) must match; no fabricated facts.
  - Completeness: contains the key points of the gold answer.
The gold answer is authoritative. Output strictly one line in this format, score being 0 (wrong) or 1 (correct): "Rating: [[score]]".

[Question]
{question}

[Gold Answer]
{gold}

[Assistant's Predicted Answer]
{gen}
"""

FAILURE_PROMPT = """You will classify the failure of an incorrect answer into exactly ONE of:
  - retrieval failure: the needed table/text was never retrieved.
  - sql failure: the SQL was wrong, errored, or returned the wrong cells.
  - reasoning failure: right data retrieved but wrong computation/interpretation.
  - parsing failure: the underlying table data was missing/garbled.
Output only the category words.

[Question]
{question}
[Gold Answer]
{gold}
[Predicted Answer]
{gen}
"""


def score_answer(question: str, gold: str, gen: str) -> int:
    raw = llm_gateway.chat_text([{"role": "user", "content": JUDGE_PROMPT.format(
        question=question, gold=gold, gen=gen)}])
    m = re.search(r"\[\[\s*([01])\s*\]\]", raw)
    if m:
        return int(m.group(1))
    m = re.search(r"\b([01])\b", raw)
    return int(m.group(1)) if m else 0


def classify_failure(question: str, gold: str, gen: str) -> Optional[str]:
    raw = llm_gateway.chat_text([{"role": "user", "content": FAILURE_PROMPT.format(
        question=question, gold=gold, gen=gen)}])
    for cat in ("retrieval failure", "sql failure", "reasoning failure", "parsing failure"):
        if cat in raw.lower():
            return cat
    return raw.strip()[:60] or None
