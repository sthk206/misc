"""
Builds data/gold_tables.json: hand-verified, clean versions of ONLY the 4 tables the
16 benchmark questions depend on. Used by the Option-B sensitivity run (TableRAG fed
correct tables) so we can separate "parser damage" from "method value".

Values are transcribed from the source PDF text (the same text used to author
benchmark_questions.json), NOT from the auto-parser. Column meanings are written
explicitly in header_context so NL->SQL can map c1..cN unambiguously -- that clarity
is the whole point of the clean condition.

Conventions: values in the units stated per table; parenthesized source values are
negative; em-dash / blank -> null. Generic row labels (e.g. "Swaps") are qualified
with their group (e.g. "Interest rate contracts - Swaps") so rows are unambiguous.

PRINCIPLE: each gold table mirrors the FULL source table (every data row, including
subtotal/reconciling/total rows), NOT just the cells the benchmark questions happen to
touch. Tailoring the gold data to the questions would leak the benchmark into the
"clean" condition and could leave tables internally inconsistent (e.g. a subtotal and
total that no longer reconcile).
"""

from __future__ import annotations

import json
import os

from poc_eval.ingestion.table_parser import to_markdown

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_PATH = os.path.join(ROOT, "poc_eval", "data", "gold_tables.json")

N = None  # null cell

# --- p207 Notional amount of derivative contracts (in billions): c1=2025, c2=2024 ---
NOTIONAL = {
    "table_id": "gold_p207_notional",
    "pdf_page": 207, "printed_page": 205, "section": "derivatives_note",
    "title": "Notional amount of derivative contracts",
    "header_context": [
        "Notional amount of free-standing derivative contracts outstanding, in $ billions.",
        "Columns: c1 = December 31 2025, c2 = December 31 2024.",
    ],
    "rows": [
        ("Interest rate contracts - Swaps", [19056, 20437]),
        ("Interest rate contracts - Futures and forwards", [3305, 3067]),
        ("Interest rate contracts - Written options", [3775, 3067]),
        ("Interest rate contracts - Purchased options", [3400, 3089]),
        ("Total interest rate contracts", [29536, 29660]),
        ("Credit derivatives", [1381, 1191]),
        ("Foreign exchange contracts - Cross-currency swaps", [5476, 4509]),
        ("Foreign exchange contracts - Spot, futures and forwards", [8187, 7005]),
        ("Foreign exchange contracts - Written options", [979, 1015]),
        ("Foreign exchange contracts - Purchased options", [953, 984]),
        ("Total foreign exchange contracts", [15595, 13513]),
        ("Equity contracts - Swaps", [1147, 850]),
        ("Equity contracts - Futures and forwards", [196, 206]),
        ("Equity contracts - Written options", [1118, 914]),
        ("Equity contracts - Purchased options", [971, 788]),
        ("Total equity contracts", [3432, 2758]),
        ("Commodity contracts - Swaps", [189, 148]),
        ("Commodity contracts - Spot, futures and forwards", [270, 191]),
        ("Commodity contracts - Written options", [119, 137]),
        ("Commodity contracts - Purchased options", [120, 125]),
        ("Total commodity contracts", [698, 601]),
        ("Total derivative notional amounts", [50642, 47723]),
    ],
}

# --- p138 Total VaR (in millions): c1-3 = 2025 Avg/Min/Max, c4-6 = 2024 Avg/Min/Max ---
VAR = {
    "table_id": "gold_p138_var",
    "pdf_page": 138, "printed_page": 136, "section": "market_risk_var",
    "title": "Total VaR",
    "header_context": [
        "Risk Management VaR at 95% confidence, in $ millions.",
        "Columns: c1=2025 Average, c2=2025 Minimum, c3=2025 Maximum, "
        "c4=2024 Average, c5=2024 Minimum, c6=2024 Maximum. NM (not meaningful) -> null.",
    ],
    "rows": [
        ("Fixed income", [35, 27, 51, 34, 26, 53]),
        ("Foreign exchange", [9, 6, 15, 15, 7, 23]),
        ("Equities", [17, 7, 138, 8, 4, 15]),
        ("Commodities and other", [24, 10, 48, 8, 6, 13]),
        ("Diversification benefit to CIB trading VaR", [-51, N, N, -32, N, N]),
        ("CIB trading VaR", [34, 21, 142, 33, 27, 42]),
        ("Credit Portfolio VaR", [21, 16, 27, 22, 18, 28]),
        ("Diversification benefit to CIB VaR", [-18, N, N, -16, N, N]),
        ("CIB VaR", [37, 23, 133, 39, 27, 52]),
        ("CCB VaR", [4, 2, 7, 3, 1, 6]),
        ("AWM VaR", [9, 8, 12, 9, 5, 10]),
        ("Corporate VaR", [10, 9, 12, 23, 7, 102]),
        ("Diversification benefit to other VaR", [-11, N, N, -10, N, N]),
        ("Other VaR", [12, 10, 14, 25, 10, 101]),
        ("Diversification benefit to CIB and other VaR", [-9, N, N, -17, N, N]),
        ("Total VaR", [40, 25, 136, 47, 30, 91]),
    ],
}

# --- p122 Wholesale credit exposure by industry (2025, in millions), 9 columns ---
CREDIT = {
    "table_id": "gold_p122_credit",
    "pdf_page": 122, "printed_page": 120, "section": "wholesale_credit_exposure",
    "title": "Wholesale credit exposure - industries (December 31, 2025)",
    "header_context": [
        "Wholesale credit exposure by industry as of December 31, 2025, in $ millions.",
        "Columns: c1=Credit exposure, c2=Investment-grade, c3=Noncriticized, "
        "c4=Criticized performing, c5=Criticized nonperforming, "
        "c6=30+ days past due and accruing loans, c7=Net charge-offs/(recoveries), "
        "c8=Credit derivative/notes hedges, c9=Liquid securities/collateral held against "
        "derivative receivables. Parenthesized = negative; blank -> null.",
        "Source column groups: c2-c5 break down credit exposure as Investment-grade (c2) vs. "
        "Noninvestment-grade (Noncriticized c3 / Criticized performing c4 / Criticized "
        "nonperforming c5); c6-c9 fall under the 'Selected metrics' group header.",
        "Row structure: named industries + 'All other' sum to 'Subtotal'; Subtotal + 'Loans "
        "held-for-sale and loans at fair value' + 'Receivables from customers' = 'Total'.",
    ],
    "rows": [
        ("Real Estate", [224858, 155712, 57478, 9967, 1701, 959, 380, -99, N]),
        ("Individuals and Individual Entities", [167700, 138142, 28677, 460, 421, 1012, -15, N, N]),
        ("Asset Managers", [152848, 117426, 35113, 304, 5, 105, 1, -5, -10626]),
        ("Consumer & Retail", [133945, 63523, 62382, 7425, 615, 115, 234, -311, N]),
        ("Technology, Media & Telecommunications", [97816, 44373, 42507, 10135, 801, 37, 281, -1078, N]),
        ("Industrials", [80606, 44078, 33166, 3101, 261, 470, 18, -68, N]),
        ("Banks & Finance Companies", [75653, 41904, 32826, 903, 20, 16, 8, -574, -657]),
        ("Healthcare", [72218, 48888, 19713, 3059, 558, 12, 191, -67, N]),
        ("Utilities", [39005, 24840, 12519, 1254, 392, 1, 63, -203, N]),
        ("Oil & Gas", [36497, 21825, 14076, 347, 249, 52, 48, -51, N]),
        ("Automotive", [35984, 19602, 15397, 958, 27, 109, 3, -277, N]),
        ("State & Municipal Govt", [32484, 31372, 1100, 3, 9, 30, N, -3, N]),
        ("Insurance", [25031, 17511, 7352, 168, N, 6, N, -20, -8310]),
        ("Chemicals & Plastics", [23790, 11251, 10355, 2091, 93, 2, 82, -239, N]),
        ("Transportation", [20861, 11450, 9097, 285, 29, 11, -3, -135, N]),
        ("Metals & Mining", [17767, 7459, 9883, 406, 19, 22, 4, -39, -67]),
        ("Central Govt", [15164, 14666, 245, 44, 209, 8, N, -1258, -1273]),
        ("Securities Firms", [7966, 4372, 3593, N, 1, 1, N, -13, -2458]),
        ("Financial Markets Infrastructure", [5734, 5306, 358, 70, N, N, N, N, N]),
        ("All other", [180171, 148214, 29887, 1953, 117, 3, 303, -19458, -5500]),
        ("Subtotal", [1446098, 971914, 425724, 42933, 5527, 2971, 1598, -23898, -28891]),
        ("Loans held-for-sale and loans at fair value", [51007, N, N, N, N, N, N, N, N]),
        ("Receivables from customers", [47336, N, N, N, N, N, N, N, N]),
        ("Total", [1544441, N, N, N, N, N, N, N, N]),
    ],
}

# --- p214 Cumulative fair value hedging adjustments (in millions), 8 columns ---
HEDGE = {
    "table_id": "gold_p214_hedge",
    "pdf_page": 214, "printed_page": 212, "section": "derivatives_note",
    "title": "Cumulative amount of fair value hedging adjustments in carrying amount of hedged items",
    "header_context": [
        "Cumulative fair value hedge basis adjustments, in $ millions.",
        "Columns: c1=2025 Carrying amount of hedged items, c2=2025 Active hedging "
        "relationships, c3=2025 Discontinued hedging relationships, c4=2025 Total, "
        "c5=2024 Carrying amount, c6=2024 Active, c7=2024 Discontinued, c8=2024 Total. "
        "Parenthesized = negative; blank -> null.",
    ],
    "rows": [
        ("Investment securities - AFS", [255109, 3693, -1374, 2319, 203141, -1675, -1959, -3634]),
        ("Long-term debt", [222611, 232, -8689, -8457, 211288, -3711, -9332, -13043]),
        ("Beneficial interests issued by consolidated VIEs", [5884, 37, N, 37, 5312, -30, -5, -35]),
    ],
}

GOLD = [NOTIONAL, VAR, CREDIT, HEDGE]


def build() -> list[dict]:
    out = []
    for t in GOLD:
        rows = [{"label": lbl, "values": vals, "raw_cells": [str(v) for v in vals]}
                for lbl, vals in t["rows"]]
        ncol = max(len(r["values"]) for r in rows)
        tbl = {
            "table_id": t["table_id"],
            "pdf_page": t["pdf_page"], "printed_page": t["printed_page"],
            "section": t["section"], "title": t["title"],
            "header_context": t["header_context"], "rows": rows,
            "n_rows": len(rows), "n_value_cols": ncol,
            "parser": "gold_verified",
        }
        tbl["markdown"] = to_markdown(tbl)
        out.append(tbl)
    return out


def main() -> None:
    tables = build()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(tables, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(tables)} verified gold tables -> {OUT_PATH}")
    for t in tables:
        print(f"  {t['table_id']}: {t['n_rows']} rows x {t['n_value_cols']} cols  ({t['title'][:50]})")


if __name__ == "__main__":
    main()
