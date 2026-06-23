# TableRAG vs. Baseline RAG — POC Summary Report (Option B: clean tables)

> ⚠️ **MOCK RUN** — generated with the offline fake gateway. Answers are canned and embeddings are bag-of-words hashes, so **all numbers below are plumbing artifacts, not findings.** Re-run against the real gateway for meaningful results.

> 🧪 **Option B (sensitivity run).** TableRAG was fed the **verified gold tables**, not the auto-parsed ones — this isolates retrieval/reasoning from parser quality. Compare against the Option A report (`summary_report.md`): the A→B gap on TableRAG is the damage attributable to PDF table parsing.

## Benchmark Overview

- **Question:** Does TableRAG provide a measurable advantage over baseline RAG when answering questions that depend on structured financial tables?
- **Document:** JPMorgan Chase & Co. 2025 Form 10-K (`corp-10k-2025.pdf`, 410 pages).
- **Questions:** 16, hand-authored with ground truth verified against the source PDF.
- **Categories:** A = Exact retrieval, B = Comparison, C = Arithmetic / aggregation, D = Cross-table / multi-step.
- **Document sections used:** Wholesale credit exposure by industry (Credit Risk); Total VaR / Market Risk (pp. 133–144); Notional amount of derivative contracts & cumulative fair value hedging adjustments (Derivatives note, pp. 205–216).
- **Systems:** Both share the same gateway LLM, embedding model, chunking (1000/200), and FAISS top-k retrieval. The only difference is TableRAG's structured-table store + NL→SQL + iterative sub-query decomposition vs. the baseline's text-only single-shot retrieve-then-generate.

## Quantitative Results

### Answer correctness (final answer right?)
| Category | Baseline RAG | TableRAG |
| --- | --- | --- |
| Exact retrieval (A) | 0/5 (0%) | 0/5 (0%) |
| Comparison (B) | 0/4 (0%) | 0/4 (0%) |
| Arithmetic / aggregation (C) | 0/4 (0%) | 0/4 (0%) |
| Cross-table / multi-step (D) | 0/3 (0%) | 0/3 (0%) |
| **Overall** | **0/16 (0%)** | **0/16 (0%)** |

### Numeric correctness (key number exact, within 1%)
| Category | Baseline RAG | TableRAG |
| --- | --- | --- |
| Exact retrieval (A) | 0/5 (0%) | 0/5 (0%) |
| Comparison (B) | 0/4 (0%) | 0/4 (0%) |
| Arithmetic / aggregation (C) | 0/4 (0%) | 0/4 (0%) |
| Cross-table / multi-step (D) | 0/3 (0%) | 0/3 (0%) |
| **Overall** | **0/16 (0%)** | **0/16 (0%)** |

### Evidence correctness (correct supporting table/page retrieved)
| Category | Baseline RAG | TableRAG |
| --- | --- | --- |
| Exact retrieval (A) | 3/5 (60%) | 4/5 (80%) |
| Comparison (B) | 4/4 (100%) | 2/4 (50%) |
| Arithmetic / aggregation (C) | 2/4 (50%) | 3/4 (75%) |
| Cross-table / multi-step (D) | 3/3 (100%) | 3/3 (100%) |
| **Overall** | **12/16 (75%)** | **12/16 (75%)** |

### Source correctness (cited the correct page)
| Category | Baseline RAG | TableRAG |
| --- | --- | --- |
| Exact retrieval (A) | 0/5 (0%) | 4/5 (80%) |
| Comparison (B) | 0/4 (0%) | 2/4 (50%) |
| Arithmetic / aggregation (C) | 0/4 (0%) | 3/4 (75%) |
| Cross-table / multi-step (D) | 0/3 (0%) | 3/3 (100%) |
| **Overall** | **0/16 (0%)** | **12/16 (75%)** |

### Head-to-head
- TableRAG correct, Baseline wrong: 0 — none
- Baseline correct, TableRAG wrong: 0 — none
- Both correct: 0 — none
- Both wrong: 16 — A1, A2, A3, A4, A5, B1, B2, B3, B4, C1, C2, C3, C4, D1, D2, D3

### Failure analysis (by type)
| Failure type | Baseline RAG | TableRAG |
| --- | --- | --- |
| retrieval failure | 4 | 4 |
| table parsing failure | 0 | 0 |
| row/column association failure | 0 | 0 |
| arithmetic failure | 0 | 0 |
| reasoning failure | 12 | 12 |
| hallucination | 0 | 0 |

## Qualitative Analysis

- **Where baseline RAG succeeded:** typically exact-retrieval (A) and comparison (B) questions where the answer number sits verbatim in a retrieved text chunk and a strong LLM can read it off.
- **Where baseline RAG failed:** see the "Baseline wrong" set above — most often arithmetic/aggregation (C) and cross-table (D), where the needed values are scattered across the linearized table text and the model must both find and compute over them in one shot.
- **Where TableRAG provided clear value:** the "TableRAG correct, Baseline wrong" set — questions answered by an explicit SQL computation (sums, ratios, max-over-rows) over the structured store.
- **Where TableRAG did not help (or hurt):** the "Baseline correct, TableRAG wrong" set — usually traceable to table-parsing noise (Option A: TableRAG is fed the auto-parsed tables) or an NL→SQL mismatch, captured in the failure-type table.

## Conclusion

On this POC, TableRAG and baseline RAG perform comparably overall (0% vs 0%). Any difference is within the noise of a small (n=16) benchmark, so this POC does NOT support a strong claim that TableRAG provides a meaningful advantage for this document. Differences on specific categories (esp. arithmetic) are worth a larger follow-up.

### Caveats
- Small sample (n=16); no statistical significance testing.
- **Option A**: TableRAG consumes auto-parsed tables, so its results include table-parsing error — a faithful end-to-end measurement, but parser quality is a confound.
- SQLite stands in for MySQL; embeddings come from the gateway rather than local bge-m3; no neural reranker on either side. These keep the comparison fair and runnable but diverge from the paper's exact stack.
- Ground truth was verified from the source PDF text, independent of the parser.
