"""
sql_tool.py

POC adaptation: the original `get_excel_rag_response_plain` POSTed to the offline Flask
NL2SQL service (DeepSeek + MySQL). The gateway/no-server environment forces this in-process,
but the LOGIC is the repo's: it reuses the offline service's exact NL2SQL prompts, the same
`extract_sql_statement` regex, and the same "load schema json -> NL2SQL -> execute SQL ->
return {sql_str, sql_execution_result, nl2sql_prompt}" flow (offline .../src/service.py).
Only swapped: DeepSeek -> gateway LLM, MySQL -> SQLite (via SQLAlchemy), HTTP -> direct call.

Configure once at startup:  sql_tool.configure(sqlite_path, schema_dir)
"""

import os
import re
import json
import logging
from sqlalchemy import create_engine, text

from poc_eval.common import llm_gateway

# main.py imports `logger` via `from tools.sql_tool import *` (the original module defined one).
logger = logging.getLogger("tablerag_poc")
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)

# --- verbatim from offline_data_ingestion_and_query_interface/src/prompt.py ---
NL2SQL_SYSTEM_PROMPT = ("You are an expert in SQL and can generate SQL statements based on "
                        "table schemas and query requirements. Respond as concisely as "
                        "possible, providing only the SQL statement without any additional "
                        "explanations.")
NL2SQL_USER_PROMPT = '''{schema_list}
Based on the schemas above, please use MySQL syntax to solve the following problem:
{user_query}
Please wrap the generated SQL statement with ```sql ```, and warp table name and each column name metioned in sql with ``, for example: ```sql SELECT `name` FROM `student_sheet1` WHERE `age` > '15';```
'''

_ENGINE = None
_SCHEMA_DIR = None


def configure(sqlite_path: str, schema_dir: str) -> None:
    global _ENGINE, _SCHEMA_DIR
    _ENGINE = create_engine(f"sqlite:///{sqlite_path}")
    _SCHEMA_DIR = schema_dir


# --- verbatim from common_utils.transfer_name ---
def transfer_name(original_name):
    import hashlib
    name = original_name.split('.')[0]
    name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    name = re.sub(r'_+', '_', name)
    if len(name) > 2:
        name = name.strip('_')
    name = name.lower()
    if name[0].isdigit():
        name = 't_' + name
    if len(name) > 64:
        prefix = name[:20].rstrip('_')
        name = f"{prefix}_{hashlib.md5(name.encode('utf-8')).hexdigest()[:8]}"
    return name


# --- verbatim from offline .../src/service.py ---
def extract_sql_statement(resp_content):
    match = re.search(r'```sql([\s\S]*?)```', resp_content, re.DOTALL)
    if match:
        return re.sub(r'\s+', ' ', match.group(1).strip())
    return resp_content


def _fetchall(sql: str) -> str:
    """Mirror of SQL_Alchemy_Helper.fetchall: SELECT -> JSON string, truncated to 1000."""
    with _ENGINE.connect() as conn:
        rows = conn.execute(text(sql)).fetchall()
        out = json.dumps([dict(r._mapping) for r in rows], ensure_ascii=False, default=str)
        return out[:1000]


def get_excel_rag_response_plain(table_name_list=None, query=None, repo_id=None):
    """In-process re-implementation of offline service.process_tablerag_request."""
    schema_list = []
    for table_name in (table_name_list or []):
        path = os.path.join(_SCHEMA_DIR, f"{transfer_name(table_name)}.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                schema_list.append(json.load(f))

    nl2sql_prompt = NL2SQL_USER_PROMPT.format(
        schema_list=json.dumps(schema_list, ensure_ascii=False), user_query=query)
    resp_content = llm_gateway.chat_text([
        {"role": "system", "content": NL2SQL_SYSTEM_PROMPT},
        {"role": "user", "content": nl2sql_prompt},
    ])
    sql_str = extract_sql_statement(resp_content)
    try:
        sql_execution_result = _fetchall(sql_str)
    except Exception as e:  # noqa: BLE001
        sql_execution_result = f"SQL execution failed: {e}"

    return {
        "query": query,
        "nl2sql_prompt": nl2sql_prompt,
        "nl2sql_response": resp_content,
        "sql_str": sql_str,
        "sql_execution_result": sql_execution_result,
    }
