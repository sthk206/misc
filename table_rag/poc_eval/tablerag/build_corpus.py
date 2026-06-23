"""
Build the corpus the repo's TableRAG consumes, from our PDF-derived JSON (no xlsx).

For each table (tables.json or gold_tables.json) we create, using the REPO's own
schema-generation logic (offline data_persistent: infer_and_convert / pandas_to_mysql_dtype
/ get_schema_and_data, copied verbatim below, with dtype_mapping imported):
  - excel_dir/<table_id>.json  : {table_name, columns, data}  -> retriever markdown + main.py
  - schema_dir/<sqlname>.json  : {table_name, column_list[[col,type,samples]], ...} -> NL2SQL
  - a SQLite table <sqlname>    : the rows, for SQL execution
Text pages (pages.json) -> doc_dir/<page>.json key-value docs (the repo's D side).

Our PDF tables have a row label + generic numeric columns (no clean headers like HybridQA's
xlsx), so columns are row_label/c1..cN and we attach the table title + header_context to the
schema as `column_context` so NL2SQL can map columns -- a documented adaptation.
"""

from __future__ import annotations

import json
import os
import random
import warnings

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from tools.sql_tool import transfer_name  # repo's name normalizer (online_inference on path)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# verbatim from offline_data_ingestion_and_query_interface/src/dtype_mapping.py
# (inlined to avoid putting the offline src dir on sys.path, which shadows online `prompt.py`)
INTEGER_DTYPE_MAPPING = {np.int8: 'TINYINT', np.int16: 'SMALLINT', np.int32: 'INT', np.int64: 'BIGINT'}
SPECIAL_INTEGER_DTYPE_MAPPING = {'Int64': 'BIGINT', 'UInt64': 'BIGINT UNSIGNED'}
FLOAT_DTYPE_MAPPING = {np.float16: 'FLOAT', np.float32: 'FLOAT', np.float64: 'DOUBLE'}
OTHER_DTYPE_MAPPING = {'boolean': 'BOOLEAN', 'datetime': 'DATETIME', 'timedelta': 'TIME',
                       'string': 'VARCHAR(255)', 'category': 'VARCHAR(255)', 'default': 'TEXT'}


# ---- verbatim from offline_data_ingestion_and_query_interface/src/data_persistent.py ----
def infer_and_convert(series):
    try:
        return pd.to_numeric(series, downcast='integer')
    except (ValueError, TypeError):
        pass
    try:
        return pd.to_numeric(series, downcast='float')
    except (ValueError, TypeError):
        pass
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return pd.to_datetime(series)
    except (ValueError, TypeError):
        pass
    return series


def pandas_to_mysql_dtype(dtype):
    if pd.api.types.is_integer_dtype(dtype):
        if str(dtype) in SPECIAL_INTEGER_DTYPE_MAPPING:
            return SPECIAL_INTEGER_DTYPE_MAPPING[str(dtype)]
        return INTEGER_DTYPE_MAPPING.get(dtype, 'INT')
    elif pd.api.types.is_float_dtype(dtype):
        return FLOAT_DTYPE_MAPPING.get(dtype, 'FLOAT')
    elif pd.api.types.is_datetime64_any_dtype(dtype):
        return OTHER_DTYPE_MAPPING['datetime']
    elif pd.api.types.is_string_dtype(dtype):
        return OTHER_DTYPE_MAPPING['string']
    return OTHER_DTYPE_MAPPING['default']


def get_sample_values(series):
    valid = [str(x) for x in series.dropna().unique() if pd.notnull(x) and len(str(x)) < 64]
    return random.sample(valid, min(3, len(valid))) if valid else ['no sample values available']


def get_schema_and_data(df):
    column_list = []
    for col in df.columns:
        column_list.append([col, pandas_to_mysql_dtype(df[col].dtype),
                            'sample values:' + str(get_sample_values(df[col]))])
    return column_list
# ---------------------------------------------------------------------------------------


def _table_to_df(t: dict) -> tuple[pd.DataFrame, list[str]]:
    ncol = t["n_value_cols"]
    columns = ["row_label"] + [f"c{i+1}" for i in range(ncol)]
    data = []
    for r in t["rows"]:
        vals = list(r["values"]) + [None] * (ncol - len(r["values"]))
        data.append([r["label"]] + vals[:ncol])
    return pd.DataFrame(data, columns=columns), columns


def build(tables: list[dict], pages: list[dict], out_dir: str) -> dict:
    excel_dir = os.path.join(out_dir, "tables")     # JSON tables (the repo's "excel_dir")
    doc_dir = os.path.join(out_dir, "docs")
    schema_dir = os.path.join(out_dir, "schema")
    for d in (excel_dir, doc_dir, schema_dir):
        os.makedirs(d, exist_ok=True)
    sqlite_path = os.path.join(out_dir, "tables.sqlite")
    if os.path.exists(sqlite_path):
        os.remove(sqlite_path)
    engine = create_engine(f"sqlite:///{sqlite_path}")

    for t in tables:
        table_name = t["table_id"]
        df, columns = _table_to_df(t)
        df_conv = df.apply(infer_and_convert)

        sqlname = transfer_name(table_name)
        df_conv.to_sql(sqlname, engine, index=False, if_exists="replace")

        schema_dict = {
            "table_name": sqlname,
            "column_list": get_schema_and_data(df_conv),
            # adaptation: our generic c-columns need their meaning spelled out for NL2SQL
            "title": t.get("title", ""),
            "column_context": t.get("header_context", []),
        }
        with open(os.path.join(schema_dir, f"{sqlname}.json"), "w", encoding="utf-8") as f:
            json.dump(schema_dict, f, ensure_ascii=False)

        # JSON table for the retriever + main.construct_initial_prompt
        with open(os.path.join(excel_dir, f"{table_name}.json"), "w", encoding="utf-8") as f:
            json.dump({"table_name": table_name, "columns": columns,
                       "data": json.loads(df.to_json(orient="values"))}, f, ensure_ascii=False)

    for pg in pages:
        with open(os.path.join(doc_dir, f"page_{pg['pdf_page']}.json"), "w", encoding="utf-8") as f:
            json.dump({f"PDF page {pg['pdf_page']}": pg["text"]}, f, ensure_ascii=False)

    return {"excel_dir": excel_dir, "doc_dir": doc_dir,
            "schema_dir": schema_dir, "sqlite_path": sqlite_path}
