"""
Version 2 -- "perfect" parser.

Models the assumption that our table parser is flawless: instead of running an
extractor, we hand-transcribe the four tables the benchmark actually tests against,
exactly as they appear in corp-10k-2025.pdf, into clean, well-typed DataFrames.
Parenthesised figures -> negative numbers; "NM"/"--" (nil / not-meaningful) -> None.

These are the *upper bound* on what table extraction could give the downstream
TableRAG system. Values were transcribed from the PDF text, and the benchmark's
ground truth was independently hand-verified against the same PDF (see
benchmark metadata `ground_truth_note`), so this is not grading the parser against
itself.

Tables (by benchmarked pdf page):
  p207 -> notional_amount_of_derivative_contracts
  p138 -> total_var
  p122 -> wholesale_credit_exposure_industries
  p214 -> fair_value_hedging_adjustments
"""
from __future__ import annotations

from typing import List, Tuple

import pandas as pd

N = None  # nil / NM


def _notional() -> pd.DataFrame:
    cols = ["category", "item", "notional_2025_usd_bn", "notional_2024_usd_bn"]
    rows = [
        ["Interest rate contracts", "Swaps", 19056, 20437],
        ["Interest rate contracts", "Futures and forwards", 3305, 3067],
        ["Interest rate contracts", "Written options", 3775, 3067],
        ["Interest rate contracts", "Purchased options", 3400, 3089],
        ["Interest rate contracts", "Total interest rate contracts", 29536, 29660],
        ["Credit derivatives", "Credit derivatives", 1381, 1191],
        ["Foreign exchange contracts", "Cross-currency swaps", 5476, 4509],
        ["Foreign exchange contracts", "Spot, futures and forwards", 8187, 7005],
        ["Foreign exchange contracts", "Written options", 979, 1015],
        ["Foreign exchange contracts", "Purchased options", 953, 984],
        ["Foreign exchange contracts", "Total foreign exchange contracts", 15595, 13513],
        ["Equity contracts", "Swaps", 1147, 850],
        ["Equity contracts", "Futures and forwards", 196, 206],
        ["Equity contracts", "Written options", 1118, 914],
        ["Equity contracts", "Purchased options", 971, 788],
        ["Equity contracts", "Total equity contracts", 3432, 2758],
        ["Commodity contracts", "Swaps", 189, 148],
        ["Commodity contracts", "Spot, futures and forwards", 270, 191],
        ["Commodity contracts", "Written options", 119, 137],
        ["Commodity contracts", "Purchased options", 120, 125],
        ["Commodity contracts", "Total commodity contracts", 698, 601],
        ["Total", "Total derivative notional amounts", 50642, 47723],
    ]
    return pd.DataFrame(rows, columns=cols)


def _total_var() -> pd.DataFrame:
    cols = [
        "metric",
        "var_2025_avg", "var_2025_min", "var_2025_max",
        "var_2024_avg", "var_2024_min", "var_2024_max",
    ]
    rows = [
        ["Fixed income", 35, 27, 51, 34, 26, 53],
        ["Foreign exchange", 9, 6, 15, 15, 7, 23],
        ["Equities", 17, 7, 138, 8, 4, 15],
        ["Commodities and other", 24, 10, 48, 8, 6, 13],
        ["Diversification benefit to CIB trading VaR", -51, N, N, -32, N, N],
        ["CIB trading VaR", 34, 21, 142, 33, 27, 42],
        ["Credit Portfolio VaR", 21, 16, 27, 22, 18, 28],
        ["Diversification benefit to CIB VaR", -18, N, N, -16, N, N],
        ["CIB VaR", 37, 23, 133, 39, 27, 52],
        ["CCB VaR", 4, 2, 7, 3, 1, 6],
        ["AWM VaR", 9, 8, 12, 9, 5, 10],
        ["Corporate VaR", 10, 9, 12, 23, 7, 102],
        ["Diversification benefit to other VaR", -11, N, N, -10, N, N],
        ["Other VaR", 12, 10, 14, 25, 10, 101],
        ["Diversification benefit to CIB and other VaR", -9, N, N, -17, N, N],
        ["Total VaR", 40, 25, 136, 47, 30, 91],
    ]
    return pd.DataFrame(rows, columns=cols)


def _wholesale_credit() -> pd.DataFrame:
    cols = [
        "industry", "credit_exposure", "investment_grade",
        "noninvestmentgrade_noncriticized", "criticized_performing",
        "criticized_nonperforming", "credit_30days_pastdue_accruing",
        "net_chargeoffs_recoveries", "credit_derivative_notes",
        "liquid_securities_collateral",
    ]
    rows = [
        ["Real Estate", 224858, 155712, 57478, 9967, 1701, 959, 380, -99, N],
        ["Individuals and Individual Entities", 167700, 138142, 28677, 460, 421, 1012, -15, N, N],
        ["Asset Managers", 152848, 117426, 35113, 304, 5, 105, 1, -5, -10626],
        ["Consumer & Retail", 133945, 63523, 62382, 7425, 615, 115, 234, -311, N],
        ["Technology, Media & Telecommunications", 97816, 44373, 42507, 10135, 801, 37, 281, -1078, N],
        ["Industrials", 80606, 44078, 33166, 3101, 261, 470, 18, -68, N],
        ["Banks & Finance Companies", 75653, 41904, 32826, 903, 20, 16, 8, -574, -657],
        ["Healthcare", 72218, 48888, 19713, 3059, 558, 12, 191, -67, N],
        ["Utilities", 39005, 24840, 12519, 1254, 392, 1, 63, -203, N],
        ["Oil & Gas", 36497, 21825, 14076, 347, 249, 52, 48, -51, N],
        ["Automotive", 35984, 19602, 15397, 958, 27, 109, 3, -277, N],
        ["State & Municipal Govt", 32484, 31372, 1100, 3, 9, 30, N, -3, N],
        ["Insurance", 25031, 17511, 7352, 168, N, 6, N, -20, -8310],
        ["Chemicals & Plastics", 23790, 11251, 10355, 2091, 93, 2, 82, -239, N],
        ["Transportation", 20861, 11450, 9097, 285, 29, 11, -3, -135, N],
        ["Metals & Mining", 17767, 7459, 9883, 406, 19, 22, 4, -39, -67],
        ["Central Govt", 15164, 14666, 245, 44, 209, 8, N, -1258, -1273],
        ["Securities Firms", 7966, 4372, 3593, N, 1, 1, N, -13, -2458],
        ["Financial Markets Infrastructure", 5734, 5306, 358, 70, N, N, N, N, N],
        ["All other", 180171, 148214, 29887, 1953, 117, 3, 303, -19458, -5500],
        ["Subtotal", 1446098, 971914, 425724, 42933, 5527, 2971, 1598, -23898, -28891],
        ["Loans held-for-sale and loans at fair value", 51007, N, N, N, N, N, N, N, N],
        ["Receivables from customers", 47336, N, N, N, N, N, N, N, N],
        ["Total", 1544441, N, N, N, N, N, N, N, N],
    ]
    return pd.DataFrame(rows, columns=cols)


def _fair_value_hedging() -> pd.DataFrame:
    cols = [
        "balance_sheet_date", "item", "carrying_amount_of_hedged_items",
        "active_hedging_relationships", "discontinued_hedging_relationships", "total",
    ]
    rows = [
        ["December 31, 2025", "Investment securities - AFS", 255109, 3693, -1374, 2319],
        ["December 31, 2025", "Long-term debt", 222611, 232, -8689, -8457],
        ["December 31, 2025", "Beneficial interests issued by consolidated VIEs", 5884, 37, N, 37],
        ["December 31, 2024", "Investment securities - AFS", 203141, -1675, -1959, -3634],
        ["December 31, 2024", "Long-term debt", 211288, -3711, -9332, -13043],
        ["December 31, 2024", "Beneficial interests issued by consolidated VIEs", 5312, -30, -5, -35],
    ]
    return pd.DataFrame(rows, columns=cols)


def extract_perfect_tables() -> List[Tuple[str, pd.DataFrame]]:
    return [
        ("notional_amount_of_derivative_contracts", _notional()),
        ("total_var", _total_var()),
        ("wholesale_credit_exposure_industries", _wholesale_credit()),
        ("fair_value_hedging_adjustments", _fair_value_hedging()),
    ]
