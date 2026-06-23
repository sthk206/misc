"""
Thin SQLAlchemy wrapper around a per-version MySQL database.

Mirrors the repo's offline `sql_alchemy_helper.SQL_Alchemy_Helper.fetchall`: returns a
JSON string of row dicts, truncated to 1000 chars (the agent only needs a peek at the
result, and this keeps prompt size bounded -- identical to the repo's behaviour).
"""
from __future__ import annotations

import json
import uuid
from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import create_engine, text

from poc_eval import config


def _default(obj):
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, uuid.UUID):
        return str(obj)
    if isinstance(obj, bytes):
        return obj.decode(errors="replace")
    raise TypeError(f"Type {type(obj)} not serializable")


class Database:
    def __init__(self, version: str):
        self.version = version
        self.engine = create_engine(config.database_url(version), pool_pre_ping=True)

    def fetchall(self, sql: str, args=None) -> str:
        with self.engine.connect() as conn:
            result = conn.execute(text(sql), args or {})
            rows = [dict(r._mapping) for r in result.fetchall()]
        out = json.dumps(rows, ensure_ascii=False, default=_default)
        return out[:1000] if len(out) > 1000 else out
