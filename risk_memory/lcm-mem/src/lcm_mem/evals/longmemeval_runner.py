"""LongMemEval-S runner (Phase 5 primary benchmark).

Per instance: ingest every haystack session through the Phase 3 pipeline into a
fresh store, answer the question through the Phase 4 composition loop, score
with an LLM judge (through the gateway, cached). Reports per question type plus
tokens and gateway calls per query — the token-efficiency curve inputs.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from lcm_mem.common import write_results
from lcm_mem.data.longmemeval import LMEInstance, load_longmemeval
from lcm_mem.encoder.embed import get_embedder
from lcm_mem.llm import prompts
from lcm_mem.llm.gateway import CachedGateway
from lcm_mem.memory.compose import ComposeConfig, answer_query
from lcm_mem.memory.ingest import canonicalize_entities, ingest_text
from lcm_mem.memory.store import MemoryStore


def _session_text(session: list[dict]) -> str:
    return "\n".join(f"{t.get('role', '?')}: {t.get('content', '')}" for t in session)


def judge_answer(
    gateway: CachedGateway, judge_model: str, question: str, gold: str, response: str
) -> bool:
    out = gateway.chat(
        [{"role": "user", "content": prompts.QA_JUDGE_V1.format(
            question=question, gold=gold, response=response)}],
        model=judge_model,
    ).strip().lower()
    return out.startswith("correct")


def run_instance(
    inst: LMEInstance,
    gateway: CachedGateway,
    compose_cfg: ComposeConfig,
    embedder_name: str,
    extract_model: str,
    db_dir: str | Path = "cache/lme_stores",
) -> dict:
    store = MemoryStore(
        Path(db_dir) / f"{inst.question_id}.sqlite", get_embedder(embedder_name)
    )
    calls_before = gateway.usage.calls
    for si, session in enumerate(inst.haystack_sessions):
        ingest_text(
            store, _session_text(session), gateway, extract_model,
            session_id=f"{inst.question_id}:{si}",
        )
    canonicalize_entities(store, gateway, extract_model)
    ingest_calls = gateway.usage.calls - calls_before

    res = answer_query(store, inst.question, gateway, compose_cfg)
    return {
        "question_id": inst.question_id,
        "question_type": inst.question_type,
        "answer": res.answer,
        "gold": inst.answer,
        "llm_calls_query": res.llm_calls,
        "llm_calls_ingest": ingest_calls,
        "derived": len(res.derived_ids),
        "reused_derived": len(res.reused_derived_ids),
        "stopped_because": res.stopped_because,
    }


def run_longmemeval(
    gateway: CachedGateway,
    compose_cfg: ComposeConfig,
    data_path: str | Path | None = None,
    embedder_name: str = "intfloat/e5-base-v2",
    extract_model: str = "gpt-4o-mini",
    judge_model: str = "gpt-4o-mini",
    limit: int | None = None,
    results_dir: str | Path = "results",
) -> dict:
    instances = load_longmemeval(data_path)
    if limit:
        instances = instances[:limit]
    rows = []
    for inst in instances:
        row = run_instance(inst, gateway, compose_cfg, embedder_name, extract_model)
        row["correct"] = judge_answer(
            gateway, judge_model, inst.question, inst.answer, row["answer"]
        )
        rows.append(row)

    by_type: dict[str, list] = defaultdict(list)
    for r in rows:
        by_type[r["question_type"]].append(r)
    metrics = {
        "overall_accuracy": sum(r["correct"] for r in rows) / max(len(rows), 1),
        "per_type": {
            t: {
                "accuracy": sum(r["correct"] for r in rs) / len(rs),
                "n": len(rs),
                "mean_query_calls": sum(r["llm_calls_query"] for r in rs) / len(rs),
                "mean_reused_derived": sum(r["reused_derived"] for r in rs) / len(rs),
            }
            for t, rs in by_type.items()
        },
        "gateway_usage": gateway.usage.as_dict(),
        "n": len(rows),
    }
    write_results(
        "longmemeval",
        config={"compose": compose_cfg.__dict__, "embedder": embedder_name,
                "extract_model": extract_model, "judge_model": judge_model,
                "limit": limit},
        metrics=metrics,
        extra={"rows": rows},
        results_dir=Path(results_dir),
    )
    return metrics
