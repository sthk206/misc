"""
Ingest a version's `.xlsx` tables into its own MySQL database.

Equivalent to the repo's offline `data_persistent.parse_excel_file_and_insert_to_db`,
but pointed at poc_eval's per-version dataset + database and tolerant of the messy
auto-extracted tables (a failed table is logged and skipped rather than aborting).
The schema JSON was already written at build time, so this step only loads rows.

Usage:
  python -m poc_eval.ingest.mysql_ingest --version perfect
"""
from __future__ import annotations

import argparse
import os

import pandas as pd
from sqlalchemy import create_engine, text
from tqdm import tqdm

from poc_eval import config
from poc_eval.parse.table_utils import clean_columns, infer_and_convert


def ensure_database(version: str) -> None:
    server = create_engine(config.server_url())
    db = config.database_name(version)
    with server.connect() as conn:
        conn.execute(text(
            f"CREATE DATABASE IF NOT EXISTS `{db}` CHARACTER SET {config.DB_CHARSET}"
        ))
        conn.commit()
    server.dispose()


def ingest(version: str) -> dict:
    ensure_database(version)
    engine = create_engine(config.database_url(version))
    xdir = config.excel_dir(version)
    files = [f for f in os.listdir(xdir) if f.endswith(".xlsx")]
    loaded, failed = 0, 0
    for fname in tqdm(files, desc=f"ingest-{version}"):
        table_name = fname[: -len(".xlsx")]
        try:
            df = pd.read_excel(os.path.join(xdir, fname))
            df = clean_columns(df.apply(infer_and_convert))
            df.to_sql(table_name, engine, index=False, if_exists="replace",
                      chunksize=1000, method="multi")
            loaded += 1
        except Exception as e:
            print(f"  skip {table_name}: {e}")
            failed += 1
    engine.dispose()
    summary = {"version": version, "loaded": loaded, "failed": failed,
               "database": config.database_name(version)}
    print(f"[mysql_ingest] {summary}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=["auto", "perfect"], required=True)
    args = ap.parse_args()
    ingest(args.version)
