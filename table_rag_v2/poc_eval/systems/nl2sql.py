"""
NL2SQL service: natural-language subquery -> SQL -> execute against MySQL.

A faithful port of the repo's offline `service.process_tablerag_request`: same prompts
(NL2SQL_SYSTEM_PROMPT / NL2SQL_USER_PROMPT), same ```sql``` extraction, same schema
JSON, same result dict keys (sql_str / sql_execution_result / nl2sql_prompt) that the
TableRAG agent reads. The only swap is the LLM call (gateway instead of a hardcoded
deepseek endpoint) and execution against the per-version poc DB.
"""
from __future__ import annotations

import json
import os
import re

from poc_eval import config
from poc_eval.common import llm_gateway
from poc_eval.parse.table_utils import transfer_name
from poc_eval.systems.db import Database
from poc_eval.systems.repo_prompts import NL2SQL_SYSTEM_PROMPT, NL2SQL_USER_PROMPT


def extract_sql_statement(resp_content: str) -> str:
    match = re.search(r"```sql([\s\S]*?)```", resp_content, re.DOTALL)
    if match:
        return re.sub(r"\s+", " ", match.group(1).strip())
    return resp_content


class NL2SQLService:
    def __init__(self, version: str):
        self.version = version
        self.db = Database(version)

    def run(self, table_name_list, query: str) -> dict:
        schema_list = []
        for table_name in table_name_list:
            name = transfer_name(table_name)
            schema_path = os.path.join(config.schema_dir(self.version), f"{name}.json")
            if os.path.exists(schema_path):
                with open(schema_path, encoding="utf-8") as f:
                    schema_list.append(json.load(f))

        nl2sql_prompt = NL2SQL_USER_PROMPT.format(
            schema_list=json.dumps(schema_list, ensure_ascii=False),
            user_query=query,
        )
        resp_content = llm_gateway.chat_text([
            {"role": "system", "content": NL2SQL_SYSTEM_PROMPT},
            {"role": "user", "content": nl2sql_prompt},
        ])
        sql_str = extract_sql_statement(resp_content)
        try:
            sql_execution_result = self.db.fetchall(sql_str)
        except Exception as e:
            sql_execution_result = f"SQL execution failed: {e}"

        return {
            "query": query,
            "schema": json.dumps(schema_list, ensure_ascii=False),
            "nl2sql_prompt": nl2sql_prompt,
            "nl2sql_response": resp_content,
            "sql_str": sql_str,
            "sql_execution_result": sql_execution_result,
        }
