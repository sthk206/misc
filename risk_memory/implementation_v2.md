# Project: Latent Composition Memory (LCM-Mem)

Research implementation plan. Hand this file to Claude Code as the project spec.

## 0. Context for the implementing agent

We are building and evaluating a memory system for LLM agents with two novel components:

1. **A latent composition predictor**: a small network `f(emb(fact_a), emb(fact_b), emb(query)) -> predicted_embedding` that predicts where the *inference combining two facts* would land in embedding space, WITHOUT running an LLM. Used as a cheap best-first-search heuristic to decide which fact pairs are worth sending to an LLM for actual composition. Trained JEPA-style on multi-hop QA decompositions (MuSiQue).
2. **A provenance-tracked memory store**: derived (LLM-inferred) facts are persisted with parent pointers to their source facts, epistemic type labels, and confidence. When a source fact is contradicted/updated, invalidation propagates through the provenance DAG.

The project is structured so that Phase 1 is a cheap "kill experiment" that can falsify the core scientific hypothesis before we invest in the full system.

### Hardware / environment constraints
- Apple M1 Max, 64 GB unified memory, corporate laptop.
- All local training uses PyTorch with `device="mps"` (fall back to CPU per-op where MPS lacks kernels; set `PYTORCH_ENABLE_MPS_FALLBACK=1`).
- Local models must be small: sentence encoder <= ~350M params (e5-base-v2 or bge-base-en-v1.5, 110M, are the defaults). Predictor is a small MLP/transformer (< 20M params). NO local LLM fine-tuning is required for the critical path.
- All generative LLM calls go through a pre-existing corporate gateway object with an OpenAI-style interface. Assume it is importable and constructed like:

```python
from company_gateway import LLMGateway  # placeholder; adjust to the real import
client = LLMGateway(config)             # exposes client.chat.completions.create(...)
resp = client.chat.completions.create(
    model="gpt-4o-mini",                 # model names are config-driven, never hardcode
    messages=[{"role": "user", "content": "..."}],
    temperature=0.0,
)
text = resp.choices[0].message.content
```

- Build a thin adapter (`src/llm/gateway.py`) wrapping this with: (a) disk cache keyed by SHA256 of (model, messages, temperature, tools) in SQLite so all experiments are reproducible and cheap to re-run; (b) retry with exponential backoff; (c) a token/cost counter logged per experiment; (d) a `dry_run` mode returning cache-only results.
- Embeddings are computed locally with `sentence-transformers` (never via the gateway) so the encoder can be fine-tuned.

### Repo layout

```
lcm-mem/
  pyproject.toml            # deps: torch, sentence-transformers, datasets, faiss-cpu (or hnswlib),
                            # numpy, pandas, scikit-learn, networkx, pydantic, typer, rich, pytest, wandb (optional, offline mode)
  configs/                  # yaml configs per experiment; every run takes exactly one config
  src/
    llm/gateway.py          # cached gateway adapter
    llm/prompts.py          # ALL prompts live here as versioned constants
    data/musique.py         # download + parse + triple extraction
    data/hard_negatives.py  # date/negation/entity-swap generation
    data/longmemeval.py     # benchmark loader
    encoder/embed.py        # batch embedding w/ MPS, caching to .npy keyed by (model_rev, text_hash)
    encoder/finetune.py     # contrastive fine-tuning (Phase 2)
    predictor/model.py      # composition predictor architectures
    predictor/train.py
    predictor/baselines.py  # mean-pool, weighted-pool, cross-encoder
    memory/store.py         # SQLite fact store + vector index
    memory/ingest.py        # LLM extraction pipeline
    memory/provenance.py    # DAG, epistemic types, invalidation propagation
    memory/compose.py       # best-first composition search at query time
    eval/kill_experiment.py
    eval/longmemeval_runner.py
    eval/ablations.py
  tests/
  notebooks/                # analysis only, no logic
  results/                  # every run writes a json + the config that produced it
```

### Global engineering rules
- Every experiment is a CLI command (`typer`) taking a config path; results JSON includes git hash, config, metrics, token spend.
- Deterministic seeds everywhere; embedding cache makes reruns fast.
- Unit tests for: triple extraction from MuSiQue, hard-negative generators (assert the negative differs from positive in exactly the intended way), provenance invalidation propagation (property-based: invalidating a leaf invalidates all and only its descendants), and the search loop termination.
- Never store raw gateway API keys in the repo; gateway config comes from env.

---

## Phase 0: Setup and data (est. 1-2 days)

1. Scaffold repo, gateway adapter with cache, embedding utility with cache.
2. Download MuSiQue (via HuggingFace `datasets`, `dgslibisey/MuSiQue` or official repo), 2WikiMultiHopQA (secondary), LongMemEval (GitHub release; use the -S variant).
3. **Triple extraction from MuSiQue** (`data/musique.py`): for each 2-hop answerable example, produce:
   - `fact_a`: declarative form of hop-1 (sub-question + answer rewritten as a statement),
   - `fact_b`: declarative form of hop-2,
   - `composed_gt`: declarative form of the full question + answer,
   - `query`: the original question text.
   Rewriting Q+A into declarative statements is done with ONE cached gateway call per item using a fixed prompt in `prompts.py` (batch, temperature 0). Also handle 3-4 hop items by emitting chained pairs: (f1,f2)->bridge12, (bridge12,f3)->bridge123, etc. Keep the unanswerable-variant items separately as hard negatives at the pair level.
   Target: >= 15k training triples, 1k val, 1k test (split by question id, no leakage of shared single-hop components across splits — split on the underlying single-hop question ids).
4. Compute and cache embeddings of all facts with frozen `intfloat/e5-base-v2` (use the "passage: " / "query: " prefixes correctly).

Deliverable: `make data` produces `data/processed/musique_triples.{train,val,test}.jsonl` and cached embeddings.

## Phase 1: KILL EXPERIMENT — can composition be predicted in latent space? (est. 3-5 days)

**Hypothesis**: `f(emb(a), emb(b), emb(q))` predicts `emb(composed_gt)` better than trivial pooling, especially when the composed fact is lexically distant from its parents.

### Models
- `MeanPool` baseline: (emb(a)+emb(b))/2, renormalized.
- `LearnedPool` baseline: scalar-weighted sum, weights learned.
- `MLPPredictor`: concat[emb_a, emb_b, emb_q, emb_a*emb_b, |emb_a-emb_b|] -> MLP (2-4 layers, hidden 1024-2048, GELU, layernorm) -> d-dim output, L2-normalized.
- `AttnPredictor`: 2-4 layer transformer over token sequence [CLS, emb_a, emb_b, emb_q] with learned type embeddings; output = CLS projection. (~5-15M params.)
- `CrossEncoderBaseline` (for the cost/quality frontier, not the latent hypothesis): ms-marco-MiniLM cross-encoder scoring (query, fact_a+fact_b concatenation); measures whether a rerank-style scorer beats the predictor at similar latency.

### Training
- Loss: primary = 1 - cosine(pred, emb(composed_gt)) with frozen encoder (targets are fixed -> no collapse possible). Add InfoNCE variant: in-batch negatives = other composed targets, PLUS mined hard negatives (see Phase 2 generators, usable here already). Run both losses; report both.
- Optimizer AdamW, lr 1e-4 (predictor), batch 256 (embeddings precomputed, so batches are cheap), early stop on val retrieval metric. Trains in minutes-to-hours on MPS/CPU since inputs are precomputed vectors.
- Query-conditioning ablation: train each predictor with and without emb_q as input.

### Evaluation (this defines success/failure — implement exactly)
Build a retrieval pool per test item: the true composed_gt embedding + 999 distractors (other composed facts + corrupted variants). Metrics:
1. **R@1, R@10, MRR** of retrieving composed_gt given pred.
2. **Lexical-overlap stratification**: bucket test items by max token-Jaccard(composed_gt, fact_a or fact_b) into terciles. Report all metrics per tercile. THE KEY CELL is the low-overlap tercile.
3. **Discrimination test**: for each item, pool = {composed_gt, date-swapped, negated, entity-swapped variants of composed_gt} (generated in Phase 2 style). Report accuracy of ranking the true one first. This measures whether the latent space even encodes correctness-relevant detail.
4. **Downstream proxy — pair pruning**: on held-out MuSiQue questions with 20 retrieved candidate facts (mix of gold supporting + distractor paragraph facts), score all pairs with each method, measure recall of the gold pair in top-k pairs for k in {1,3,5,10}, and latency per 1000 pairs.

### Decision gate (write results to `results/phase1_verdict.md`)
- **STRONG PASS**: predictor beats MeanPool by >= 10 MRR points overall AND >= 10 in the low-overlap tercile AND matches/beats cross-encoder recall@5 for pair pruning at >= 10x lower latency. -> proceed full plan.
- **WEAK PASS (filter regime)**: predictor achieves >= 90% gold-pair recall@10 while pruning >= 90% of pairs, even if per-item MRR gains are modest. -> proceed, but position predictor as high-recall filter; cross-encoder reranks the survivors.
- **FAIL**: neither. -> proceed with Phases 3-5 using cross-encoder or LLM-scored pruning instead of the predictor; the paper pivots to provenance/invalidation + an analysis of WHY latent composition fails (the discrimination-test numbers are the analysis).

## Phase 2: Encoder hard-negative fine-tuning (est. 3-4 days, only if Phase 1 >= weak pass)

Goal: fix embedding insensitivity to dates/negation/entities, improving both the predictor targets and the memory system's retrieval.

1. `data/hard_negatives.py` generators over composed facts (rule-based + one cached LLM pass to fluency-check):
   - date/number swap (regex-detect dates/numbers, perturb),
   - negation insertion/removal (LLM-assisted, verified by a template check),
   - entity swap (swap with another entity of same type from the corpus; use spaCy NER locally),
   - role swap where applicable ("A hired B" -> "B hired A").
2. Contrastive fine-tune e5-base with MultipleNegativesRankingLoss + explicit hard negatives (sentence-transformers supports this natively; MPS-compatible; batch 32-64, 1-3 epochs, lr 2e-5). ~110M params trains comfortably in 64GB.
3. **Regression guard**: evaluate before/after on (a) a slice of MTEB retrieval (e.g., NFCorpus, SciFact — small, local) to ensure general retrieval doesn't degrade > 2 points, (b) the Phase 1 discrimination test (should improve a lot).
4. Retrain Phase 1 predictor on the new encoder; if unfreezing the encoder jointly with the predictor, targets must come from an EMA copy of the encoder with stop-gradient (BYOL-style, momentum 0.99-0.999). Compare frozen-new-encoder vs joint-EMA; keep whichever wins on the Phase 1 metrics.

## Phase 3: Memory store with provenance (est. 1 week; independent of Phase 1 outcome — build in parallel if desired)

### Data model (SQLite + vector index)
```
facts(id, text, embedding_id, type ENUM('observed','derived','world_bridge'),
      confidence REAL, created_at, valid_from, invalidated_at NULL,
      source_session, extraction_model, depth INT)
provenance(child_id, parent_id)          # DAG edges; observed facts have no parents
entities(id, name, canonical_id)         # canonical_id after alias resolution
fact_entities(fact_id, entity_id)
contradictions(fact_id, contradicted_by_fact_id, detected_at)
```
Vector index: FAISS flat (corpus will be small; exact search) over ALL non-invalidated facts of every type — collapsed-tree style, one flat index, no level routing.

### Ingestion (`memory/ingest.py`)
Per session/message batch, one gateway call with a fixed extraction prompt producing JSON: atomic declarative facts + entities per fact + salience 1-10. Store as `observed` with confidence 1.0, depth 0. Entity canonicalization: embed entity names, cluster by cosine > threshold + LLM tie-break for ambiguous merges (cached).

### Invalidation (`memory/provenance.py`)
- On ingesting a fact, check for contradiction against existing facts sharing >= 1 entity: candidate set by entity overlap + embedding similarity > 0.75, then ONE batched gateway call classifying pairs as {contradicts, updates, duplicates, unrelated}.
- `updates`/`contradicts` -> set `invalidated_at` on the old fact, record in `contradictions`, then **propagate**: BFS over provenance descendants; each derived descendant is marked `stale` (soft-invalid: excluded from retrieval, kept for possible recomputation). Recomputation is lazy: only when a stale fact would have been retrieved for a live query do we re-run its composition with surviving parents (or drop it).
- Property-based tests (hypothesis lib): random DAGs, invalidate random leaf, assert exactly the descendant closure goes stale.

## Phase 4: Query-time composition search (est. 1 week)

`memory/compose.py`, the core loop:

1. Query -> embed; gateway call extracts query entities (cached prompt).
2. **Candidate facts**: union of (top-20 dense retrieval) and (entity-linked facts: facts sharing canonical entities with the query; if graph is large later, replace with Personalized PageRank a la HippoRAG via networkx).
3. **Answerability check**: cheap heuristic first (does any single fact embedding exceed sim threshold t_ans to query?), else one gateway call: "answerable from these facts alone? yes/no". If yes -> answer, done, no composition.
4. **Best-first search over compositions**:
   - Frontier = candidate facts. Score all pairs with the predictor: priority = cosine(f(emb_a, emb_b, emb_q), emb_q).
   - Pop best pair -> ONE gateway call to verbalize the composition ("Given fact A and fact B, state the most relevant inference for query Q, or output NONE"; also return a self-rated confidence and whether external world knowledge was used).
   - If non-NONE: store as `derived` (or `world_bridge` if world knowledge flagged), confidence = llm_conf * decay^depth (decay ~0.85), parents = {a, b}, add to frontier and to the vector index.
   - Repeat until: answerability check passes, OR budget exhausted (max L LLM calls, default 5), OR max depth D (default 3), OR all pair scores < threshold.
   - Answer from the final retrieved+derived set with citations to fact ids.
5. **Persistence is the point**: derived facts survive across queries. Log cache-hit style metrics: fraction of queries answered using previously derived facts, LLM calls saved.

Config-switchable pair scorers (this IS the ablation surface): `predictor | mean_pool | cross_encoder | llm_score | random`.

## Phase 5: Evaluation and ablations (est. 1-2 weeks)

### Primary: LongMemEval-S
- Runner ingests each instance's sessions through Phase 3 pipeline, answers questions through Phase 4, scores with the benchmark's official evaluator (LLM-judge calls go through the gateway, cached).
- Report per question type (multi-hop, temporal, knowledge-update, single-session, abstention) — the taxonomy maps directly onto our claims: predictor -> multi-hop/temporal; invalidation -> knowledge-update.
- Also report tokens per query and gateway calls per query. Token-efficiency curves (accuracy vs LLM-call budget L in {1,3,5,10}) are a headline figure.

### Baselines
1. Full-context stuffing (upper-bound reference where context fits),
2. Dense RAG top-k,
3. Entity-boosted RAG (our candidate stage without composition) — approximates Mem0-style,
4. Iterative retrieval (Self-Ask style loop, same LLM budget L) WITHOUT persistence — the critical "same compute, no materialization" control,
5. Full system with each pair scorer variant.

### Ablation grid (each toggled from the full system, same seeds)
- predictor -> mean_pool / cross_encoder / llm_score
- persistence ON/OFF (OFF = derived facts discarded after each query)
- invalidation ON/OFF (OFF -> measure poisoning: knowledge-update accuracy should crater)
- query-conditioning of predictor ON/OFF
- confidence decay ON/OFF
- fine-tuned encoder (Phase 2) vs stock e5

### Secondary experiment (cheap, high-value): "retrieval in inference space"
On MuSiQue test: index all corpus facts; ALSO index predictor outputs for top-M entity-linked pairs per question. Measure whether querying against predicted-composition vectors surfaces gold supporting pairs that plain dense retrieval misses. This isolates the predictor's value for RETRIEVAL independent of the whole memory system.

## Milestone summary

| Milestone | Gate |
|---|---|
| M0 Phase 0 done | triples extracted, embeddings cached |
| M1 Kill experiment verdict | STRONG / WEAK / FAIL written to results/ |
| M2 Encoder fine-tune | discrimination test improves, MTEB slice within 2 pts |
| M3 Memory store | property tests green, ingestion of a full LongMemEval instance < N gateway calls |
| M4 Composition loop | end-to-end answer with citations on toy corpus + logged reuse metrics |
| M5 LongMemEval results + ablation grid | full results JSONs + plots |

## Known risks and mitigations
- MPS op gaps: enable fallback env var; all heavy training uses precomputed embeddings so CPU fallback is tolerable.
- Gateway rate limits/cost: aggressive caching, batch prompts, `dry_run` CI mode; log spend per run; cap per-experiment budget in config.
- MuSiQue declarative rewriting noise: manually audit 50 random triples early; iterate prompt before bulk run.
- Contradiction detector false positives nuking good facts: require entity overlap AND classifier confidence; keep soft-invalidation (stale) reversible.
- Benchmark leakage: never let LongMemEval data touch predictor/encoder training.

## Explicit non-goals (do not build)
- No local LLM fine-tuning of the decoder/reasoner (original "idea 2" training scheme is dropped per design discussion — gradient path through sampled text is broken; latent predictor replaces it).
- No UI. CLI + JSON results only.
- No multi-user/production concerns (auth, concurrency).
