"""
Generates results/summary_report.md from the evaluation rows. Re-run automatically by
run_eval.py, so the numbers always reflect the latest run.
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Any

ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
RESULTS_DIR = os.path.join(ROOT, "poc_eval", "results")


def _report_path(clean: bool) -> str:
    name = "summary_report_optionB.md" if clean else "summary_report.md"
    return os.path.join(RESULTS_DIR, name)

CATEGORIES = [("A", "Exact retrieval"), ("B", "Comparison"),
              ("C", "Arithmetic / aggregation"), ("D", "Cross-table / multi-step")]
SYSTEMS = [("baseline_rag", "Baseline RAG"), ("tablerag", "TableRAG")]


def _acc(rows: list[dict], system: str, metric: str, category: str | None = None) -> tuple[int, int]:
    sel = [r for r in rows if r["system"] == system and (category is None or r["category"] == category)]
    if metric == "numeric_correct":
        sel = [r for r in sel if r[metric] != ""]
    correct = sum(int(r[metric]) for r in sel if r[metric] != "")
    return correct, len(sel)


def _pct(c: int, n: int) -> str:
    return f"{c}/{n} ({100*c/n:.0f}%)" if n else "-"


def _accuracy_table(rows: list[dict], metric: str) -> str:
    head = "| Category | " + " | ".join(label for _, label in SYSTEMS) + " |"
    sep = "| --- | " + " | ".join("---" for _ in SYSTEMS) + " |"
    lines = [head, sep]
    for cat, label in CATEGORIES:
        cells = [_pct(*_acc(rows, sysid, metric, cat)) for sysid, _ in SYSTEMS]
        lines.append(f"| {label} ({cat}) | " + " | ".join(cells) + " |")
    overall = [_pct(*_acc(rows, sysid, metric)) for sysid, _ in SYSTEMS]
    lines.append("| **Overall** | " + " | ".join(f"**{c}**" for c in overall) + " |")
    return "\n".join(lines)


def _by_id(rows: list[dict]) -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in rows:
        out[r["question_id"]][r["system"]] = r
    return out


def generate_report(rows: list[dict[str, Any]], mock: bool = False, clean: bool = False) -> None:
    byid = _by_id(rows)
    n_q = len(byid)

    tr_wins = [qid for qid, d in byid.items()
               if d.get("tablerag", {}).get("answer_correct") == 1
               and d.get("baseline_rag", {}).get("answer_correct") == 0]
    bl_wins = [qid for qid, d in byid.items()
               if d.get("baseline_rag", {}).get("answer_correct") == 1
               and d.get("tablerag", {}).get("answer_correct") == 0]
    both_ok = [qid for qid, d in byid.items()
               if d.get("tablerag", {}).get("answer_correct") == 1
               and d.get("baseline_rag", {}).get("answer_correct") == 1]
    both_fail = [qid for qid, d in byid.items()
                 if d.get("tablerag", {}).get("answer_correct") == 0
                 and d.get("baseline_rag", {}).get("answer_correct") == 0]

    bl_c, bl_n = _acc(rows, "baseline_rag", "answer_correct")
    tr_c, tr_n = _acc(rows, "tablerag", "answer_correct")
    bl_overall = bl_c / bl_n if bl_n else 0
    tr_overall = tr_c / tr_n if tr_n else 0

    # Failure breakdown
    fails: dict[str, dict[str, int]] = {s: defaultdict(int) for s, _ in SYSTEMS}
    for r in rows:
        if r["failure_type"]:
            fails[r["system"]][r["failure_type"]] += 1

    # Conservative conclusion
    diff = tr_overall - bl_overall
    tr_arith = _acc(rows, "tablerag", "answer_correct", "C")
    bl_arith = _acc(rows, "baseline_rag", "answer_correct", "C")
    tr_cross = _acc(rows, "tablerag", "answer_correct", "D")
    bl_cross = _acc(rows, "baseline_rag", "answer_correct", "D")
    arith_adv = (tr_arith[0] / tr_arith[1] if tr_arith[1] else 0) - \
                (bl_arith[0] / bl_arith[1] if bl_arith[1] else 0)
    if diff >= 0.15 or arith_adv >= 0.25:
        verdict = (
            f"On this POC, TableRAG shows a measurable advantage over baseline RAG "
            f"(overall answer accuracy {tr_overall:.0%} vs {bl_overall:.0%}), concentrated "
            f"in arithmetic/aggregation and cross-table questions where structured SQL "
            f"execution helps. Given the small sample (n={n_q}), this is suggestive rather "
            f"than conclusive."
        )
    elif diff <= -0.10:
        verdict = (
            f"On this POC, TableRAG did NOT outperform baseline RAG (overall {tr_overall:.0%} "
            f"vs {bl_overall:.0%}); the added table parsing/SQL machinery introduced errors "
            f"that outweighed its benefits on this document. n={n_q}."
        )
    else:
        verdict = (
            f"On this POC, TableRAG and baseline RAG perform comparably overall "
            f"({tr_overall:.0%} vs {bl_overall:.0%}). Any difference is within the noise of a "
            f"small (n={n_q}) benchmark, so this POC does NOT support a strong claim that "
            f"TableRAG provides a meaningful advantage for this document. Differences on "
            f"specific categories (esp. arithmetic) are worth a larger follow-up."
        )

    mock_banner = (
        "> ⚠️ **MOCK RUN** — generated with the offline fake gateway. Answers are canned and "
        "embeddings are bag-of-words hashes, so **all numbers below are plumbing artifacts, "
        "not findings.** Re-run against the real gateway for meaningful results.\n\n"
        if mock else ""
    )
    if clean:
        option_banner = (
            "> 🧪 **Option B (sensitivity run).** TableRAG was fed the **verified gold tables**, "
            "not the auto-parsed ones — this isolates retrieval/reasoning from parser quality. "
            "Compare against the Option A report (`summary_report.md`): the A→B gap on TableRAG "
            "is the damage attributable to PDF table parsing.\n\n"
        )
        title = "# TableRAG vs. Baseline RAG — POC Summary Report (Option B: clean tables)"
    else:
        option_banner = (
            "> Option A (realistic end-to-end): TableRAG consumes the auto-parsed tables; parser "
            "errors count against it. See `summary_report_optionB.md` for the clean-table "
            "sensitivity run if present.\n\n"
        )
        title = "# TableRAG vs. Baseline RAG — POC Summary Report"

    md = f"""{title}

{mock_banner}{option_banner}## Benchmark Overview

- **Question:** Does TableRAG provide a measurable advantage over baseline RAG when answering questions that depend on structured financial tables?
- **Document:** JPMorgan Chase & Co. 2025 Form 10-K (`corp-10k-2025.pdf`, 410 pages).
- **Questions:** {n_q}, hand-authored with ground truth verified against the source PDF.
- **Categories:** A = Exact retrieval, B = Comparison, C = Arithmetic / aggregation, D = Cross-table / multi-step.
- **Document sections used:** Wholesale credit exposure by industry (Credit Risk); Total VaR / Market Risk (pp. 133–144); Notional amount of derivative contracts & cumulative fair value hedging adjustments (Derivatives note, pp. 205–216).
- **Systems:** Both share the same gateway LLM, embedding model, chunking (1000/200), and FAISS top-k retrieval. The only difference is TableRAG's structured-table store + NL→SQL + iterative sub-query decomposition vs. the baseline's text-only single-shot retrieve-then-generate.

## Quantitative Results

### Answer correctness (final answer right?)
{_accuracy_table(rows, "answer_correct")}

### Numeric correctness (key number exact, within 1%)
{_accuracy_table(rows, "numeric_correct")}

### Evidence correctness (correct supporting table/page retrieved)
{_accuracy_table(rows, "evidence_correct")}

### Source correctness (cited the correct page)
{_accuracy_table(rows, "source_correct")}

### Head-to-head
- TableRAG correct, Baseline wrong: {len(tr_wins)} — {', '.join(tr_wins) or 'none'}
- Baseline correct, TableRAG wrong: {len(bl_wins)} — {', '.join(bl_wins) or 'none'}
- Both correct: {len(both_ok)} — {', '.join(both_ok) or 'none'}
- Both wrong: {len(both_fail)} — {', '.join(both_fail) or 'none'}

### Failure analysis (by type)
| Failure type | Baseline RAG | TableRAG |
| --- | --- | --- |
""" + "\n".join(
        f"| {ft} | {fails['baseline_rag'].get(ft, 0)} | {fails['tablerag'].get(ft, 0)} |"
        for ft in [
            "retrieval failure", "table parsing failure", "row/column association failure",
            "arithmetic failure", "reasoning failure", "hallucination",
        ]
    ) + f"""

## Qualitative Analysis

- **Where baseline RAG succeeded:** typically exact-retrieval (A) and comparison (B) questions where the answer number sits verbatim in a retrieved text chunk and a strong LLM can read it off.
- **Where baseline RAG failed:** see the "Baseline wrong" set above — most often arithmetic/aggregation (C) and cross-table (D), where the needed values are scattered across the linearized table text and the model must both find and compute over them in one shot.
- **Where TableRAG provided clear value:** the "TableRAG correct, Baseline wrong" set — questions answered by an explicit SQL computation (sums, ratios, max-over-rows) over the structured store.
- **Where TableRAG did not help (or hurt):** the "Baseline correct, TableRAG wrong" set — usually traceable to table-parsing noise (Option A: TableRAG is fed the auto-parsed tables) or an NL→SQL mismatch, captured in the failure-type table.

## Conclusion

{verdict}

### Caveats
- Small sample (n={n_q}); no statistical significance testing.
- **Option A**: TableRAG consumes auto-parsed tables, so its results include table-parsing error — a faithful end-to-end measurement, but parser quality is a confound.
- SQLite stands in for MySQL; embeddings come from the gateway rather than local bge-m3; no neural reranker on either side. These keep the comparison fair and runnable but diverge from the paper's exact stack.
- Ground truth was verified from the source PDF text, independent of the parser.
"""
    path = _report_path(clean)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(md)
    print(f"Wrote report -> {path}")
