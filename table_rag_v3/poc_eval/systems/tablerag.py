"""
TableRAG agent -- a faithful port of online_inference/main.py `_run`, instrumented to
emit a full per-question trace (retrieved table, each decomposed subquery, the docs
retrieved for it, the generated SQL + execution result, and the subquery answer) so a
reader can see exactly where a question succeeded or failed.

Loop (identical structure to the repo):
  1. retrieve the most relevant table for the question; render it to Markdown as the
     seed context (SYSTEM_EXPLORE_PROMPT);
  2. iterate up to MAX_ITER: the controller LLM either decomposes the query into one
     `solve_subquery` tool call, or emits "<Answer>: ...";
  3. each subquery is solved by retrieving supporting docs + running NL2SQL over the
     table and combining both via COMBINE_PROMPT;
  4. subquery answers are fed back as tool messages until the controller answers.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

from poc_eval import config
from poc_eval.common import llm_gateway
from poc_eval.systems.markdown import excel_to_markdown
from poc_eval.systems.nl2sql import NL2SQLService
from poc_eval.systems.repo_prompts import COMBINE_PROMPT, SYSTEM_EXPLORE_PROMPT
from poc_eval.systems.retriever import GatewayRetriever

SOLVE_SUBQUERY_TOOL = [{
    "type": "function",
    "function": {
        "name": "solve_subquery",
        "description": "Return answer for the decomposed subquery.",
        "parameters": {
            "type": "object",
            "properties": {
                "subquery": {
                    "type": "string",
                    "description": "The subquery to be solved, only take natural language as input.",
                }
            },
            "required": ["subquery"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}]


class TableRAGSystem:
    name_prefix = "tablerag"

    def __init__(self, version: str, table_hint: bool = False):
        self.version = version
        self.table_hint = table_hint
        self.name = f"tablerag-{version}" + ("-hint" if table_hint else "")
        self.retriever = GatewayRetriever(version, system_tag="tablerag")
        self.nl2sql = NL2SQLService(version)
        self.max_iter = config.MAX_ITER

    # --- helpers ----------------------------------------------------------------------
    @staticmethod
    def _table_stem(filename: str) -> str:
        return filename.replace(".xlsx", "").replace(".json", "")

    def _pick_table(self, files: List[str]) -> str:
        for f in files:
            if f.endswith(".xlsx"):
                return self._table_stem(f)
        return self._table_stem(files[0]) if files else ""

    @staticmethod
    def _extract_subqueries(response: Any) -> Tuple[str, List[str], List[str]]:
        reasoning = getattr(response, "content", "") or ""
        subqueries, ids = [], []
        tool_calls = getattr(response, "tool_calls", None)
        if tool_calls:
            for call in tool_calls:
                try:
                    subqueries.append(json.loads(call.function.arguments)["subquery"])
                    ids.append(call.id)
                except Exception:
                    continue
        return reasoning, subqueries, ids

    # --- main entry -------------------------------------------------------------------
    def answer(self, question: str, qid: str = "", table_title: str | None = None) -> Dict[str, Any]:
        trace: Dict[str, Any] = {"system": self.name, "question_id": qid,
                                 "question": question, "steps": []}

        # Table-selection query. With the hint on, append the gold table title --
        # the analog of the repo's `query + f"The given table is in {table_id}"`
        # (main.py:133). The hint affects only top-1 table selection, never the
        # per-subquery retrieval below, exactly as in the repo.
        table_query = question
        if self.table_hint and table_title:
            table_query = f"{question} The given table is in {table_title}"
        trace["table_hint"] = self.table_hint
        trace["table_query"] = table_query

        _, _, files = self.retriever.retrieve(table_query)
        table_name = self._pick_table(files)
        trace["retrieved_table"] = table_name
        trace["retrieved_files_top"] = files

        xlsx_path = os.path.join(config.excel_dir(self.version), f"{table_name}.xlsx")
        table_md = excel_to_markdown(xlsx_path) if os.path.exists(xlsx_path) else "Can NOT find table content!"
        messages: List[Any] = [{
            "role": "user",
            "content": SYSTEM_EXPLORE_PROMPT.format(query=question, table_content=table_md),
        }]

        final_answer = ""
        for it in range(self.max_iter):
            response = llm_gateway.chat(messages, tools=SOLVE_SUBQUERY_TOOL)
            reasoning, subqueries, tool_ids = self._extract_subqueries(response)
            step: Dict[str, Any] = {"iter": it + 1, "reasoning": reasoning, "subqueries": []}

            if not subqueries and "<Answer>" in (reasoning or "") and it != 0:
                final_answer = reasoning.split("<Answer>", 1)[1].lstrip(": ").strip()
                trace["steps"].append(step)
                break
            if not subqueries:
                messages.append({"role": "user", "content": "ERROR: Did not call tool with a subquery!"})
                trace["steps"].append(step)
                continue

            messages.append(response)
            for subquery, tool_id in zip(subqueries, tool_ids):
                docs, _, doc_files = self.retriever.retrieve(subquery)
                doc_content = "\n".join(list(dict.fromkeys(docs))[:3])
                sql_info = self.nl2sql.run([table_name], subquery)
                combine_prompt = COMBINE_PROMPT.format(
                    docs=doc_content,
                    schema=sql_info["schema"],
                    nl2sql_model_response=sql_info["sql_str"],
                    sql_execute_result=sql_info["sql_execution_result"],
                    query=subquery,
                )
                sub_answer = llm_gateway.chat_text([{"role": "user", "content": combine_prompt}])
                messages.append({"role": "tool", "tool_call_id": tool_id,
                                 "content": "Subquery Answer: " + (sub_answer or "")})
                step["subqueries"].append({
                    "subquery": subquery,
                    "retrieved_doc_files": doc_files,
                    "sql": sql_info["sql_str"],
                    "sql_result": sql_info["sql_execution_result"],
                    "subquery_answer": sub_answer,
                })
            trace["steps"].append(step)

        trace["final_answer"] = final_answer
        return trace
