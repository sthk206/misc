"""SQLite helpers shared by the stores. Start SQLite; schemas kept portable to
Postgres (no SQLite-only types beyond what a migration handles)."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path


def connect(db_path: str | Path) -> sqlite3.Connection:
    if str(db_path) != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def now() -> float:
    return time.time()


def dumps(obj) -> str:
    return json.dumps(obj, separators=(",", ":"), default=str)


def loads(s: str | None):
    return json.loads(s) if s else None
