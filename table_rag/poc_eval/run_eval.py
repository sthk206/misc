"""
Evaluation harness.

Runs every benchmark question through BOTH systems, scores four axes, classifies
failures, writes results/evaluation_results.csv, and regenerates
results/summary_report.md from the results.

Usage:
  python -m poc_eval.run_eval                 # real gateway (must be wired up)
  python -m poc_eval.run_eval --limit 3       # first 3 questions only
  python -m poc_eval.run_eval --mock          # offline plumbing check (fake answers)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any, Optional

from poc_eval.common import llm_gateway
from poc_eval.common.jsonutil import extract_json

ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
BENCH_PATH = os.path.join(ROOT, "poc_eval", "benchmark", "benchmark_questions.json")
RESULTS_DIR = os.path.join(ROOT, "poc_eval", "results")
GOLD_TABLES_PATH = os.path.join(ROOT, "poc_eval", "data", "gold_tables.json")


def csv_path(clean: bool) -> str:
    name = "evaluation_results_optionB.csv" if clean else "evaluation_results.csv"
    return os.path.join(RESULTS_DIR, name)

FAILURE_TYPES = [
    "retrieval failure",
    "table parsing failure",
    "row/column association failure",
    "arithmetic failure",
    "reasoning failure",
    "hallucination",
]

JUDGE_SYSTEM = (
    "You are an impartial grader. Given a question, a reference answer, and a candidate "
    "answer about a financial filing, decide if the candidate is correct (same key fact / "
    "number / entity as the reference; ignore phrasing and rounding within 1%). "
    "Reply with a single character: 1 if correct, 0 if not."
)

CLASSIFY_SYSTEM = (
    "You classify the failure of a QA system on a financial-table question into exactly "
    "one category from this list: " + "; ".join(FAILURE_TYPES) + ". "
    "Definitions: 'retrieval failure' = the correct table/page was not retrieved; "
    "'table parsing failure' = the right table was found but its cell values were garbled "
    "during extraction; 'row/column association failure' = read the wrong row/column "
    "intersection; 'arithmetic failure' = retrieved correct values but computed wrong; "
    "'reasoning failure' = wrong multi-step logic/interpretation; 'hallucination' = "
    "produced a value/claim absent from the evidence. Reply with ONLY the category text."
)


# --------------------------------------------------------------------------- scoring
def _to_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).replace(",", "").replace("$", "").replace("%", "").strip()
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()")
    try:
        f = float(s)
        return -f if neg else f
    except ValueError:
        return None


def numeric_correct(pred: Any, gold: Optional[float]) -> Optional[bool]:
    if gold is None:
        return None
    p = _to_float(pred)
    if p is None:
        return False
    if gold == 0:
        return abs(p) < 1e-6
    return abs(p - gold) / abs(gold) <= 0.01


def judge_answer(question: str, gold: str, candidate: str) -> bool:
    if not candidate or candidate.strip().upper() in ("NOT FOUND", "MOCK", "MOCK ANSWER"):
        return False
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": f"Question: {question}\nReference: {gold}\nCandidate: {candidate}"},
    ]
    out = llm_gateway.chat_text(messages).strip()
    return out.startswith("1")


def pages_hit(system_pages: list[int], gold_pages: list[int]) -> bool:
    return bool(set(system_pages or []) & set(gold_pages or []))


def classify_failure(q: dict[str, Any], res: dict[str, Any], evidence_ok: bool) -> str:
    if not evidence_ok:
        return "retrieval failure"
    evidence = json.dumps(res.get("retrieved_evidence", []))[:500]
    sql = json.dumps(res.get("sql_log", []))[:800]
    messages = [
        {"role": "system", "content": CLASSIFY_SYSTEM},
        {"role": "user", "content": (
            f"Question ({q['category']}): {q['question']}\n"
            f"Reference answer: {q['answer']}\n"
            f"System answer: {res.get('answer')} (value={res.get('value')})\n"
            f"Retrieved evidence: {evidence}\nSQL log: {sql}"
        )},
    ]
    out = llm_gateway.chat_text(messages).strip().lower()
    for ft in FAILURE_TYPES:
        if ft in out:
            return ft
    return "reasoning failure"


# ----------------------------------------------------------------------------- driver
def evaluate(limit: Optional[int] = None, clean: bool = False) -> list[dict[str, Any]]:
    from poc_eval.baseline_rag.pipeline import BaselineRAG
    from poc_eval.tablerag.runner import TableRAGRunner

    with open(BENCH_PATH) as f:
        bench = json.load(f)
    questions = bench["questions"][:limit] if limit else bench["questions"]

    print(llm_gateway.config_banner())
    mode = "Option B (TableRAG fed VERIFIED gold tables)" if clean \
        else "Option A (TableRAG fed auto-parsed tables)"
    print(f"Mode: {mode}")
    print(f"Building systems and evaluating {len(questions)} questions x 2 systems...\n")
    tr = TableRAGRunner(tables_path=GOLD_TABLES_PATH if clean else None)
    systems = [BaselineRAG(), tr]

    rows: list[dict[str, Any]] = []
    for q in questions:
        gold_val = q.get("numeric_answer")
        for sys in systems:
            res = sys.answer(q["question"])
            ev_pages = res.get("evidence_pages") or [
                e.get("pdf_page") for e in res.get("retrieved_evidence", [])
            ]
            evidence_ok = pages_hit(ev_pages, q["pdf_pages"])
            num_ok = numeric_correct(res.get("value"), gold_val)
            # Answer correctness: numeric shortcut for A/C, else LLM judge.
            if q["category"] in ("A", "C") and num_ok is True:
                ans_ok = True
            else:
                ans_ok = judge_answer(q["question"], q["answer"], str(res.get("answer", "")))
                if num_ok is True:
                    ans_ok = True
            source_ok = pages_hit(res.get("cited_pages", []), q["pdf_pages"])
            failure = "" if ans_ok else classify_failure(q, res, evidence_ok)

            rows.append({
                "question_id": q["id"],
                "category": q["category"],
                "question": q["question"],
                "gold_answer": q["answer"],
                "gold_value": gold_val,
                "gold_pages": ";".join(map(str, q["pdf_pages"])),
                "system": sys.name,
                "system_answer": res.get("answer"),
                "system_value": res.get("value"),
                "cited_pages": ";".join(map(str, res.get("cited_pages", []))),
                "evidence_pages": ";".join(map(str, ev_pages)),
                "answer_correct": int(bool(ans_ok)),
                "numeric_correct": "" if num_ok is None else int(num_ok),
                "evidence_correct": int(evidence_ok),
                "source_correct": int(source_ok),
                "failure_type": failure,
                "iterations": res.get("iterations", 1),
                "sql_used": " || ".join(
                    s.get("sql") or "" for s in res.get("sql_log", []) if s.get("sql")
                ),
            })
            print(f"  [{q['id']}/{sys.name}] ans_ok={int(bool(ans_ok))} "
                  f"num_ok={num_ok} ev_ok={int(evidence_ok)} -> {str(res.get('answer'))[:60]!r}")
    return rows


def write_csv(rows: list[dict[str, Any]], clean: bool = False) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = csv_path(clean)
    fields = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows -> {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--mock", action="store_true", help="offline plumbing check (fake LLM)")
    ap.add_argument("--clean", action="store_true",
                    help="Option B: feed TableRAG the verified gold tables (isolates parser impact)")
    args = ap.parse_args()

    if args.mock:
        from poc_eval.common import mock_gateway
        mock_gateway.install()
        print(">>> MOCK MODE: answers are fake; metrics are plumbing artifacts only.\n")

    rows = evaluate(limit=args.limit, clean=args.clean)
    write_csv(rows, clean=args.clean)

    from poc_eval.report import generate_report
    generate_report(rows, mock=args.mock, clean=args.clean)


if __name__ == "__main__":
    main()
