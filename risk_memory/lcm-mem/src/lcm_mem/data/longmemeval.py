"""LongMemEval benchmark loader (use the -S variant).

The dataset ships as a JSON list of instances. Obtain `longmemeval_s.json` from
the official release (https://github.com/xiaowu0162/LongMemEval); this loader
also tries the HuggingFace mirror `xiaowu0162/longmemeval` if no local file is
given. NEVER let this data touch predictor/encoder training (leakage rule).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class LMEInstance:
    question_id: str
    question_type: str  # multi-session, temporal-reasoning, knowledge-update, ...
    question: str
    answer: str
    question_date: str
    haystack_dates: list[str]
    haystack_sessions: list[list[dict]]  # sessions of {role, content} turns
    answer_session_ids: list[str]


def download_longmemeval(dest: str | Path = "data/raw/longmemeval_s.json") -> Path:
    dest = Path(dest)
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import hf_hub_download

    got = hf_hub_download(
        repo_id="xiaowu0162/longmemeval",
        filename="longmemeval_s.json",
        repo_type="dataset",
    )
    dest.write_bytes(Path(got).read_bytes())
    return dest


def load_longmemeval(path: str | Path | None = None) -> list[LMEInstance]:
    p = Path(path) if path else download_longmemeval()
    raw = json.loads(p.read_text())
    out = []
    for r in raw:
        out.append(
            LMEInstance(
                question_id=r["question_id"],
                question_type=r["question_type"],
                question=r["question"],
                answer=str(r["answer"]),
                question_date=r.get("question_date", ""),
                haystack_dates=r.get("haystack_dates", []),
                haystack_sessions=r.get("haystack_sessions", []),
                answer_session_ids=[str(s) for s in r.get("answer_session_ids", [])],
            )
        )
    return out
