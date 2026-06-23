"""
Shared table helpers. These mirror the repo's offline `data_persistent.py` /
`common_utils.py` so the schema JSON and table names produced here are byte-for-byte
compatible with the NL2SQL step (which reads `{table_name, column_list}` schema dicts
and emits MySQL against ``-quoted identifiers).
"""
from __future__ import annotations

import hashlib
import random
import re
import warnings
from typing import List

import pandas as pd

# MySQL dtype mapping (subset of the repo's dtype_mapping.py, enough for our data).
_INTEGER = "BIGINT"
_FLOAT = "DOUBLE"
_TEXT = "TEXT"


def transfer_name(original_name: str) -> str:
    """Identical to offline common_utils.transfer_name: filesystem/SQL-safe identifier."""
    name = str(original_name).split(".")[0]
    name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    name = re.sub(r"_+", "_", name)
    if len(name) > 2:
        name = name.strip("_")
    name = name.lower()
    if not name:
        name = "t_empty"
    if name[0].isdigit():
        name = "t_" + name
    if len(name) > 64:
        prefix = name[:20].rstrip("_")
        hash_suffix = hashlib.md5(name.encode("utf-8")).hexdigest()[:8]
        name = f"{prefix}_{hash_suffix}"
    return name


def infer_and_convert(series: pd.Series) -> pd.Series:
    """Mirror data_persistent.infer_and_convert: best-effort numeric/datetime coercion."""
    try:
        return pd.to_numeric(series, downcast="integer")
    except (ValueError, TypeError):
        pass
    try:
        return pd.to_numeric(series, downcast="float")
    except (ValueError, TypeError):
        pass
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return pd.to_datetime(series)
    except (ValueError, TypeError):
        pass
    return series


def pandas_to_mysql_dtype(dtype) -> str:
    if pd.api.types.is_integer_dtype(dtype):
        return _INTEGER
    if pd.api.types.is_float_dtype(dtype):
        return _FLOAT
    if pd.api.types.is_bool_dtype(dtype):
        return "TINYINT(1)"
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "DATETIME"
    return _TEXT


def _sample_values(series: pd.Series) -> List[str]:
    valid = [str(x) for x in series.dropna().unique() if pd.notnull(x) and len(str(x)) < 64]
    sample = random.sample(valid, min(3, len(valid))) if valid else []
    return sample if sample else ["no sample values available"]


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Mirror data_persistent.transfer_df_columns: safe, de-duplicated column names."""
    df = df.copy()
    df.columns = [transfer_name(c) for c in df.columns]
    df.columns = [
        "no" if i == 0 and (not col or pd.isna(col)) else col
        for i, col in enumerate(df.columns)
    ]
    seen: dict = {}
    new_cols = []
    for col in df.columns:
        if col in seen:
            seen[col] += 1
            new_cols.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            new_cols.append(col)
    df.columns = new_cols
    return df


def build_schema(df: pd.DataFrame, table_name: str) -> dict:
    """Produce the `{table_name, column_list:[[col, dtype, 'sample values:...'], ...]}`
    schema dict the NL2SQL step consumes (see offline service.process_tablerag_request)."""
    column_list = []
    for col in df.columns:
        column_list.append([
            col,
            pandas_to_mysql_dtype(df[col].dtype),
            "sample values:" + str(_sample_values(df[col])),
        ])
    return {"table_name": table_name, "column_list": column_list}
