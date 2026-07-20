"""ALL prompts live here as versioned constants.

Never inline prompt strings elsewhere; bump the version suffix when editing so
the gateway cache does not silently serve stale-prompt responses.
"""

DECLARATIVE_REWRITE_V1 = """\
Rewrite the question and its answer as a single short declarative statement of fact.
Keep every entity, date, and qualifier from the question and answer. Do not add
information. Output ONLY the statement.

Question: {question}
Answer: {answer}
Statement:"""

COMPOSE_BRIDGE_V1 = """\
Combine the two facts into ONE declarative statement that expresses what follows
from both together. Preserve entities, dates, and qualifiers; do not add outside
knowledge. Output ONLY the statement.

Fact 1: {fact_a}
Fact 2: {fact_b}
Combined statement:"""

FLUENCY_CHECK_V1 = """\
For each numbered sentence, answer Y if it is grammatical, fluent English and N
otherwise. Output ONLY a JSON list of "Y"/"N" strings, one per sentence, in order.

{numbered_sentences}"""

FACT_EXTRACTION_V1 = """\
Extract the atomic factual statements from the text below. For each fact list the
named entities it mentions and a salience score from 1 (trivial) to 10 (central).
Output ONLY JSON: a list of objects {{"fact": str, "entities": [str], "salience": int}}.
Each fact must be a self-contained declarative sentence (resolve pronouns).

Text:
{text}

JSON:"""

CONTRADICTION_CLASSIFY_V1 = """\
For each numbered pair of statements, classify their relation as exactly one of:
"contradicts" (they cannot both be true), "updates" (the NEW statement supersedes
the OLD one, e.g. a changed value or status), "duplicates" (same information), or
"unrelated". Output ONLY a JSON list of these strings, one per pair, in order.

{numbered_pairs}"""

ENTITY_MERGE_V1 = """\
Do these two names refer to the same real-world entity? Answer ONLY "yes" or "no".

Name 1: {name_a}
Name 2: {name_b}"""

QUERY_ENTITIES_V1 = """\
List the named entities mentioned in this query. Output ONLY a JSON list of strings.

Query: {query}"""

ANSWERABILITY_V1 = """\
Can the question be answered from these facts alone, without combining them with
outside knowledge or with each other in non-obvious ways? Answer ONLY "yes" or "no".

Facts:
{facts}

Question: {question}"""

COMPOSE_INFERENCE_V1 = """\
Given fact A and fact B, state the single most relevant inference that combines
them and helps answer the query. If no meaningful combined inference exists,
output NONE. Also rate your confidence in the inference (0.0-1.0) and say whether
you needed external world knowledge beyond the two facts.
Output ONLY JSON: {{"inference": str or "NONE", "confidence": float, "used_world_knowledge": bool}}

Fact A: {fact_a}
Fact B: {fact_b}
Query: {query}
JSON:"""

FINAL_ANSWER_V1 = """\
Answer the question using ONLY the numbered facts below. Cite the fact numbers you
used in square brackets, e.g. [2][5]. If the facts are insufficient, say
"I don't know". Be concise.

Facts:
{facts}

Question: {question}
Answer:"""

PAIR_SCORE_V1 = """\
How useful would combining these two facts be for answering the query?
Answer ONLY a number from 0 (useless) to 10 (exactly what is needed).

Fact A: {fact_a}
Fact B: {fact_b}
Query: {query}
Score:"""

QA_JUDGE_V1 = """\
Judge whether the model response correctly answers the question, given the gold
answer. Minor phrasing differences are fine; the factual content must match.
Answer ONLY "correct" or "incorrect".

Question: {question}
Gold answer: {gold}
Model response: {response}"""

NEGATION_REWRITE_V1 = """\
Rewrite the sentence to assert the OPPOSITE by inserting or removing a negation.
Change as few words as possible. Output ONLY the rewritten sentence.

Sentence: {sentence}
Rewritten:"""
