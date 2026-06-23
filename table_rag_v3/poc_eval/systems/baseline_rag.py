"""
Baseline RAG -- the "plain text RAG" control to compare TableRAG against.

Deliberately simple and table-unaware: it retrieves the top-k document-text chunks
(the narrative 10-K pages, which DO contain the table numbers as flowing text) and
asks the LLM to answer directly with page citations. No table extraction, no SQL, no
query decomposition. This isolates the value added by TableRAG's structured-table +
SQL machinery.

The system prompt contains the phrase "financial analyst answering questions" and
requests a strict JSON object, which is the contract mock_gateway routes on.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict

from poc_eval import config
from poc_eval.common import llm_gateway
from poc_eval.systems.retriever import GatewayRetriever

BASELINE_SYSTEM_PROMPT = (
    "You are a financial analyst answering questions about a company's 10-K filing. "
    "You are given excerpts retrieved from the filing. Answer the user's question using "
    "ONLY those excerpts. Respond with a strict JSON object and nothing else, of the form: "
    '{"answer": "<concise natural-language answer>", "value": <the key numeric value or null>, '
    '"pages": [<pdf page numbers you used>]}.'
)

TOP_K = 8


def _pdf_page(filename: str) -> int | None:
    m = re.search(r"page_(\d+)\.json", filename)
    return int(m.group(1)) if m else None


class BaselineRAGSystem:
    def __init__(self, version: str = "perfect"):
        # Doc text is identical across versions, so the baseline is version-independent;
        # we reuse one version's doc/ corpus and a text-only (no-tables) index.
        self.name = "baseline-rag"
        self.retriever = GatewayRetriever(version, system_tag="baseline", include_tables=False)

    def answer(self, question: str, qid: str = "", table_title: str | None = None) -> Dict[str, Any]:
        # table_title is accepted for a uniform call signature but unused: the baseline
        # is table-unaware (no table selection to hint).
        docs, _, files = self.retriever.retrieve(question, recall_num=config.RECALL_NUM, rerank_num=TOP_K)
        context = "\n\n".join(docs)
        pages = sorted({p for f in files if (p := _pdf_page(f)) is not None})

        user = f"Retrieved excerpts:\n{context}\n\nQuestion: {question}"
        raw = llm_gateway.chat_text([
            {"role": "system", "content": BASELINE_SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ])
        answer_text, value = raw, None
        try:
            parsed = json.loads(re.search(r"\{.*\}", raw, re.DOTALL).group(0))
            answer_text = parsed.get("answer", raw)
            value = parsed.get("value")
        except Exception:
            pass

        return {
            "system": self.name,
            "question_id": qid,
            "question": question,
            "retrieved_pages": pages,
            "retrieved_files_top": files,
            "raw_response": raw,
            "final_answer": answer_text if isinstance(answer_text, str) else json.dumps(answer_text),
            "value": value,
        }
