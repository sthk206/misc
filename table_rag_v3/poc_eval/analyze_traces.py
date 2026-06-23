"""
Turn a results run's traces.jsonl into a scannable per-question Markdown report.

For each question: the expected answer + source metadata (pdf/printed page, table,
supporting evidence, gold reasoning), followed by one collapsible subsection per system
showing the *full trail* of how that system reached its answer (baseline: retrieved
pages + raw response; tablerag: table selection, every subquery with its retrieved
docs, generated SQL, SQL result, and subquery answer, across all agent iterations).

Markdown (not Streamlit) on purpose: no server, scannable by scrolling, diffable, and
renders on GitHub. The long trails are wrapped in <details> so the page stays compact.

Usage:
  python -m poc_eval.analyze_traces                      # latest run -> analysis.md
  python -m poc_eval.analyze_traces --run poc_eval/results/<stamp>
  python -m poc_eval.analyze_traces --out /tmp/analysis.md
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any, Dict, List

from poc_eval import config

SYSTEM_ORDER = [
    "baseline-rag",
    "tablerag-auto",
    "tablerag-auto-hint",
    "tablerag-perfect",
    "tablerag-perfect-hint",
]


def _latest_run() -> str:
    runs = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "results", "*")))
    if not runs:
        raise SystemExit("no results found; run poc_eval.run_eval first")
    return runs[-1]


def _load(run_dir: str):
    rows = [json.loads(l) for l in open(os.path.join(run_dir, "traces.jsonl"), encoding="utf-8")]
    by_key = {(r["system"], r["id"]): r for r in rows}
    present = [s for s in SYSTEM_ORDER if any(r["system"] == s for r in rows)]
    # keep any non-standard system names too, appended in first-seen order
    for r in rows:
        if r["system"] not in present:
            present.append(r["system"])
    with open(config.BENCHMARK_FILE, encoding="utf-8") as f:
        questions = json.load(f)["questions"]
    return by_key, present, questions


def _chip(row: Dict[str, Any] | None) -> str:
    if row is None:
        return "·"
    return "✅" if row.get("score") else "❌"


def _short(system: str) -> str:
    return (system.replace("tablerag-", "")
            .replace("baseline-rag", "base")
            .replace("perfect", "perf")
            .replace("-hint", "+h"))


def _esc(s: Any) -> str:
    return str(s).replace("|", "\\|").replace("\n", " ").strip()


def _baseline_trail(trace: Dict[str, Any]) -> List[str]:
    out = [f"- **retrieved pages:** {trace.get('retrieved_pages')}",
           f"- **retrieved files:** {trace.get('retrieved_files_top')}"]
    if trace.get("value") is not None:
        out.append(f"- **extracted value:** {trace.get('value')}")
    raw = trace.get("raw_response")
    if raw:
        out += ["- **raw model response:**", "  ```json", *[f"  {l}" for l in str(raw).splitlines()], "  ```"]
    return out


def _tablerag_trail(trace: Dict[str, Any]) -> List[str]:
    out = [f"- **table hint:** {'on' if trace.get('table_hint') else 'off'}",
           f"- **selected table:** `{trace.get('retrieved_table')}`",
           f"- **top retrieved files:** {trace.get('retrieved_files_top')}"]
    for step in trace.get("steps", []):
        reasoning = (step.get("reasoning") or "").strip()
        out.append(f"- **iter {step['iter']}**" + (f" — reasoning: {reasoning}" if reasoning else ""))
        for sq in step.get("subqueries", []):
            out.append(f"  - **subquery:** {sq.get('subquery')}")
            out.append(f"    - retrieved docs: {sq.get('retrieved_doc_files')}")
            out.append(f"    - SQL: `{_esc(sq.get('sql'))}`")
            out.append(f"    - SQL result: `{_esc(str(sq.get('sql_result'))[:400])}`")
            out.append(f"    - subquery answer: {sq.get('subquery_answer')}")
    return out


def _trail(trace: Dict[str, Any]) -> List[str]:
    if not trace:
        return ["- _(no trace)_"]
    if trace.get("error"):
        return [f"- **harness error:** `{_esc(trace['error'])}`"]
    if "steps" in trace or "retrieved_table" in trace:
        return _tablerag_trail(trace)
    return _baseline_trail(trace)


def build(run_dir: str, out_path: str) -> str:
    by_key, present, questions = _load(run_dir)
    L: List[str] = [f"# Trace analysis — `{os.path.basename(run_dir)}`", ""]

    # quick per-question scoreboard
    L += ["| ID | Cat | " + " | ".join(present) + " |",
          "|---|---|" + "---|" * len(present)]
    for q in questions:
        qid = q["id"]
        if not any((s, qid) in by_key for s in present):
            continue
        cells = " | ".join(_chip(by_key.get((s, qid))) for s in present)
        L.append(f"| {qid} | {q['category']} | {cells} |")
    L.append("")

    for q in questions:
        qid = q["id"]
        if not any((s, qid) in by_key for s in present):
            continue
        scoreline = "  ".join(f"{_short(s)} {_chip(by_key.get((s, qid)))}" for s in present)
        L += ["---", "", f"## {qid} · category {q['category']}  ({scoreline})", "",
              f"**Question:** {q['question']}", "",
              f"**Expected answer:** {q['answer']}"
              + (f"  _(numeric: {q.get('numeric_answer')} {q.get('unit','')})_" if q.get("numeric_answer") is not None else ""),
              f"**Source:** pdf p{q.get('pdf_pages')} · printed p{q.get('printed_pages')} · table: \"{q.get('table_title','')}\"",
              f"**Supporting evidence:** {q.get('supporting_evidence','')}",
              f"**Gold reasoning:** {q.get('reasoning','')}", ""]

        for s in present:
            row = by_key.get((s, qid))
            if row is None:
                L += [f"### {s} — _(not run)_", ""]
                continue
            mark = "✅" if row.get("score") else "❌"
            fail = f" · failure: _{row['failure']}_" if row.get("failure") else ""
            pred = _esc(row.get("predicted") or "")
            pred_snip = (pred[:80] + "…") if len(pred) > 80 else pred
            L += [
                "<details>",
                f"<summary><b>{s}</b> — {mark}{fail} · predicted: “{pred_snip}”</summary>",
                "",
                f"**Predicted answer:** {row.get('predicted')}",
                "",
                *_trail(row.get("trace") or {}),
                "",
                "</details>",
                "",
            ]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="results dir (default: latest)")
    ap.add_argument("--out", default=None, help="output md path (default: <run>/analysis.md)")
    args = ap.parse_args()
    run_dir = args.run or _latest_run()
    out = args.out or os.path.join(run_dir, "analysis.md")
    path = build(run_dir, out)
    print(f"[analyze_traces] wrote {path}")
