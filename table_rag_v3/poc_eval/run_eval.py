"""
Run the three-system comparison over the benchmark and emit traces + a report.

Systems:
  baseline-rag       -- plain text RAG (no tables, no SQL)
  tablerag-auto      -- TableRAG over auto-extracted (pdfplumber) tables
  tablerag-perfect   -- TableRAG over hand-perfect tables

For every (system, question) it records a full trace (subqueries, retrieved files,
SQL + results, subquery answers), an LLM-judge score, and -- for wrong answers -- a
failure category. Outputs land in poc_eval/results/<timestamp>/:
  traces.jsonl   one line per (system, question) with the complete trace
  summary.json   per-system accuracy + per-category breakdown
  report.md      human-readable comparison + per-question table

Usage:
  python -m poc_eval.run_eval --mock                 # offline plumbing check
  python -m poc_eval.run_eval                         # real gateway (needs creds)
  python -m poc_eval.run_eval --mock --limit 2 --systems tablerag-perfect
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import traceback
from collections import defaultdict
from typing import Any, Dict, List

from poc_eval import config
from poc_eval.common import llm_gateway
from poc_eval.eval import judge

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
ALL_SYSTEMS = [
    "baseline-rag",
    "tablerag-auto",
    "tablerag-auto-hint",
    "tablerag-perfect",
    "tablerag-perfect-hint",
]


def _load_questions() -> List[Dict[str, Any]]:
    with open(config.BENCHMARK_FILE, encoding="utf-8") as f:
        return json.load(f)["questions"]


def _build_system(name: str):
    if name == "baseline-rag":
        from poc_eval.systems.baseline_rag import BaselineRAGSystem
        return BaselineRAGSystem()
    from poc_eval.systems.tablerag import TableRAGSystem
    specs = {
        "tablerag-auto": ("auto", False),
        "tablerag-auto-hint": ("auto", True),
        "tablerag-perfect": ("perfect", False),
        "tablerag-perfect-hint": ("perfect", True),
    }
    if name in specs:
        version, hint = specs[name]
        return TableRAGSystem(version, table_hint=hint)
    raise ValueError(name)


def run(systems: List[str], limit: int | None, mock: bool) -> str:
    if mock:
        from poc_eval.common import mock_gateway
        mock_gateway.install()
        print("[run_eval] mock gateway installed -- metrics are plumbing artifacts, not findings.")
    else:
        print(llm_gateway.config_banner())

    questions = _load_questions()
    if limit:
        questions = questions[:limit]

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(RESULTS_DIR, stamp)
    os.makedirs(out_dir, exist_ok=True)
    traces_path = os.path.join(out_dir, "traces.jsonl")

    records: List[Dict[str, Any]] = []
    with open(traces_path, "w", encoding="utf-8") as tf:
        for sys_name in systems:
            print(f"\n=== building {sys_name} ===")
            system = _build_system(sys_name)
            for q in questions:
                qid, question, gold = q["id"], q["question"], q["answer"]
                try:
                    trace = system.answer(question, qid=qid, table_title=q.get("table_title"))
                    gen = trace.get("final_answer", "") or ""
                    score = judge.score_answer(question, gold, gen)
                    failure = None if score else judge.classify_failure(question, gold, gen)
                except Exception as e:
                    trace = {"system": sys_name, "question_id": qid, "question": question,
                             "error": f"{e}", "traceback": traceback.format_exc()}
                    gen, score, failure = "", 0, "harness error"
                rec = {"system": sys_name, "id": qid, "category": q["category"],
                       "question": question, "gold": gold, "predicted": gen,
                       "score": score, "failure": failure}
                records.append(rec)
                tf.write(json.dumps({**rec, "trace": trace}, ensure_ascii=False) + "\n")
                tf.flush()
                print(f"  [{sys_name}] {qid} score={score}"
                      + (f" ({failure})" if failure else ""))

    summary = _summarize(records, systems, questions)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    report = _report(records, summary, systems, questions, mock)
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(report)
    print("\n" + report)
    print(f"\n[run_eval] wrote results to {out_dir}")
    return out_dir


def _summarize(records, systems, questions) -> Dict[str, Any]:
    cats = sorted({q["category"] for q in questions})
    summary: Dict[str, Any] = {"n_questions": len(questions), "systems": {}}
    for s in systems:
        rs = [r for r in records if r["system"] == s]
        correct = sum(r["score"] for r in rs)
        by_cat = {c: {"correct": sum(r["score"] for r in rs if r["category"] == c),
                      "total": sum(1 for r in rs if r["category"] == c)} for c in cats}
        fails: Dict[str, int] = defaultdict(int)
        for r in rs:
            if not r["score"] and r["failure"]:
                fails[r["failure"]] += 1
        summary["systems"][s] = {
            "accuracy": round(correct / len(rs), 3) if rs else 0.0,
            "correct": correct, "total": len(rs),
            "by_category": by_cat, "failures": dict(fails),
        }
    return summary


def _report(records, summary, systems, questions, mock) -> str:
    cats = sorted({q["category"] for q in questions})
    lines = ["# TableRAG poc_eval report", ""]
    if mock:
        lines += ["> **MOCK RUN** -- numbers are plumbing artifacts (canned LLM replies), "
                  "not real findings. Re-run against the real gateway for meaningful metrics.", ""]
    lines += ["## Overall accuracy", "",
              "| System | Accuracy | Correct/Total | " + " | ".join(cats) + " |",
              "|---|---|---|" + "---|" * len(cats)]
    for s in systems:
        st = summary["systems"][s]
        cat_cells = " | ".join(f"{st['by_category'][c]['correct']}/{st['by_category'][c]['total']}" for c in cats)
        lines.append(f"| {s} | {st['accuracy']:.1%} | {st['correct']}/{st['total']} | {cat_cells} |")
    lines += ["", "## Per-question scores", "",
              "| ID | Cat | " + " | ".join(systems) + " |",
              "|---|---|" + "---|" * len(systems)]
    by_q = {q["id"]: q for q in questions}
    for qid in [q["id"] for q in questions]:
        cells = []
        for s in systems:
            r = next((r for r in records if r["system"] == s and r["id"] == qid), None)
            cells.append("✅" if (r and r["score"]) else "❌")
        lines.append(f"| {qid} | {by_q[qid]['category']} | " + " | ".join(cells) + " |")
    lines += ["", "## Failure breakdown", ""]
    for s in systems:
        fails = summary["systems"][s]["failures"]
        lines.append(f"- **{s}**: " + (", ".join(f"{k}×{v}" for k, v in fails.items()) if fails else "none"))
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", default=ALL_SYSTEMS, choices=ALL_SYSTEMS)
    ap.add_argument("--limit", type=int, default=None, help="only first N questions")
    ap.add_argument("--mock", action="store_true", help="use deterministic mock gateway")
    args = ap.parse_args()
    run(args.systems, args.limit, args.mock)
