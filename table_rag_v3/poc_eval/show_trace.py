"""
Pretty-print the per-question trace across all systems from a results run, so you can
see exactly where each system succeeded or failed (subqueries, retrieved files, SQL +
results, subquery answers).

Usage:
  python -m poc_eval.show_trace --question C1
  python -m poc_eval.show_trace --question C1 --run poc_eval/results/<stamp>
"""
from __future__ import annotations

import argparse
import glob
import json
import os


def _latest_run() -> str:
    runs = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "results", "*")))
    if not runs:
        raise SystemExit("no results found; run poc_eval.run_eval first")
    return runs[-1]


def show(question_id: str, run_dir: str) -> None:
    path = os.path.join(run_dir, "traces.jsonl")
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    rows = [r for r in rows if r["id"] == question_id]
    if not rows:
        raise SystemExit(f"question {question_id} not found in {path}")
    print(f"# Question {question_id}: {rows[0]['question']}")
    print(f"# Gold: {rows[0]['gold']}\n")
    for r in rows:
        mark = "✅" if r["score"] else "❌"
        print(f"================ {r['system']}  {mark} score={r['score']}"
              + (f"  ({r['failure']})" if r["failure"] else "") + " ================")
        print(f"predicted: {r['predicted']}")
        t = r["trace"]
        if "retrieved_pages" in t:
            print(f"retrieved pages: {t['retrieved_pages']}")
        if "retrieved_table" in t:
            print(f"retrieved table: {t['retrieved_table']}")
        for step in t.get("steps", []):
            print(f"  -- iter {step['iter']} reasoning={step['reasoning'][:80]!r}")
            for sq in step["subqueries"]:
                print(f"     subquery: {sq['subquery']}")
                print(f"       docs: {sq['retrieved_doc_files']}")
                print(f"       sql : {sq['sql']}")
                print(f"       sql result: {str(sq['sql_result'])[:160]}")
                print(f"       subquery answer: {sq['subquery_answer']!r}")
        print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", required=True, help="question id, e.g. C1")
    ap.add_argument("--run", default=None, help="results dir (default: latest)")
    args = ap.parse_args()
    show(args.question, args.run or _latest_run())
