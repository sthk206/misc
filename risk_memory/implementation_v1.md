# Implementation Plan: Memory System for Risk Intelligence Agent

Target executor: Claude Code. This document is self-contained; prior design discussion is
summarized where needed. Build in the order given: **Phase A (ACE) → Phase B (Preference
store) → Phase C (Findings store)**. Phase C completes project phase 1.

---

## 0. Scope changes from earlier drafts (read first)

**DROPPED — do not implement:**
- The JEPA-style latent composition predictor (`f(emb(a), emb(b)) → ê`).
- The contrastively fine-tuned encoder (hard negatives for dates/negation/entity swaps).
- The MuSiQue "kill experiment." It existed solely to de-risk the predictor; with the
  predictor gone there is nothing for it to kill. Do not build any MuSiQue/2WikiMultiHopQA
  tooling.

**Rationale (for the record):** after entity + validity-window + hierarchy scoping, candidate
fact sets for composition are tens of items, not thousands. A single LLM reasoning pass over
the full scoped set dominates learned pairwise search at this scale, and is explainable to
model-risk validation, which a learned latent heuristic is not. The predictor becomes
relevant only if scoped candidate sets grow large (year-2 concern); the findings store we
build now would be its training data then.

**RETAINED:** provenance DAG with epistemic typing, pattern nodes with canonicalization,
upward invalidation propagation, the ACE→pattern justification edge, iterative composition
as an *escalation path* (Section C.5), quantitative attribution-coverage stopping rule.

---

## 1. System overview

Three stores, one lineage chain, plus live tools:

```
atom (trading engine)      → live MCP tools. NEVER memorized. Source of truth for numbers.
S1 ACE playbook store      → procedural memory. Compiled into context every session.
S2 Preference store        → per-manager structured profile. Injected whole at session start.
S3 Findings store          → Graphiti temporal KG (facts) + insight DAG + pattern registry.
```

Lineage chain (edges point DOWN to evidence; invalidation propagates UP):

```
ACE rule → pattern node → insight nodes (DAG) → {Zep facts, atom snapshots, other insights}
                                                  Zep facts → episodes (raw sources)
```

Single-ownership rules (enforce in code review):
- Numbers live in atom. No store caches risk numbers except as immutable snapshot
  references (snapshot_id + hash), used only for provenance.
- Preferences live ONLY in S2. The reflector (A.3) and insight writer (C.4) must not
  write preference-like content; route candidates to S2's confirmation queue.
- Conclusions live ONLY in S3's DAG. ACE entries contain no case details, only
  directives + a justification pointer.

---

## 2. Claude Agent SDK integration & the baseline

We use `claude-agent-sdk` (Python). Key primitives we rely on (verified against current
docs — re-check versions at implementation time):
- `ClaudeSDKClient` / `query()` with `ClaudeAgentOptions`.
- In-process MCP servers: `@tool` + `create_sdk_mcp_server` — all three stores are exposed
  this way (same process, no subprocess management, easy to mock in evals).
- Hooks: `Stop` / `SessionEnd` (trigger reflection + insight write-back),
  `PreCompact` (flush provenance-relevant tool results to the episode log before the SDK
  compacts them away), `PostToolUse` (log atom snapshot ids for provenance).
- Subagents (`AgentDefinition`): `reflector` (A.3), `insight_writer` (C.4),
  `investigator` (C.5 escalation loop) — keeps their token usage out of the main context.
- `setting_sources` + skills: predefined workflows (report build, drilldown, stress-PnL
  attribution) remain ordinary skills files; ACE-managed content is compiled into a
  dedicated section (A.4), not hand-edited.

### 2.1 The baseline to beat (freeze this before building anything)

`baseline/` = a stock SDK agent with: atom MCP tools, the static predefined skills,
CLAUDE.md project context, native compaction, per-session history only. **No S1/S2/S3.**
For multi-session baseline evals, prior sessions are available only as raw transcript files
the agent may grep (this is the honest "SDK out of the box" memory story).

Every eval in this plan runs `treatment vs baseline` through the same harness
(Section 6.1). A store ships only if it beats baseline on its primary metrics without
regressing token cost beyond the stated budget.

---

## Phase A — ACE playbook layer (build first)

### A.1 Data model (SQLite or Postgres; start SQLite)

```
playbook_entry:
  id, text                  # imperative directive, <= 60 tokens enforced
  scope                     # global | manager:<id> | mode:{morning,predef,adhoc}
  status                    # candidate | active | retired
  helpful_count, harmful_count, last_fired_at
  justification_ptr         # pattern_id (Phase C) | trajectory_ref | null
  created_by                # reflector | promotion | human
  created_at, approved_by, approved_at
```

Note: `justification_ptr` is nullable until Phase C exists; backfill when the pattern
registry lands. Entries created in Phase A point at trajectory refs (session ids).

### A.2 Injection path (deterministic, no retrieval)

At session start, compile all `active` entries matching (manager, mode) into a
`## Playbook` block appended to the system prompt, ordered by scope specificity then
recency. Hard budget: 2,000 tokens; if exceeded, lowest-scoring entries
(helpful−harmful, tie-break stale `last_fired_at`) are dropped and flagged for the
pruning queue. This budget is the forcing function that keeps ACE the "expensive
shortlist" — do not raise it to fix eval scores; fix curation instead.

### A.3 Write path (grow-and-refine deltas, never monolithic rewrite)

`Stop`/`SessionEnd` hook spawns the `reflector` subagent over the session transcript:
1. Emits **deltas only**: `ADD(entry)`, `INCR helpful/harmful(entry_id, evidence_span)`,
   `MERGE(entry_ids)`, `RETIRE(entry_id, reason)`.
2. ADD/MERGE/RETIRE go to the **candidate queue**; only counters apply automatically.
3. Dedup gate on ADD: embed text, cosine vs existing entries in scope; > 0.85 → convert
   to INCR on the existing entry instead.
4. Human approval: phase-1 UI is a CLI (`python -m playbook review`) listing candidates
   with evidence spans; approve/edit/reject. Nothing becomes `active` without approval.
   (This is the audit requirement, not a nice-to-have.)

### A.4 Exit criteria for Phase A

- Injection + budget enforcement + delta pipeline + CLI review working end-to-end.
- Eval suite (Section 6.2) green: adherence uplift over baseline, retention across
  sessions, no bleed across manager scopes, token overhead within budget.

---

## Phase B — Preference store (build second)

### B.1 Data model

```
preference:
  manager_id, key           # e.g. layout.chart_order, thresholds.dod_flag_ccy_pair
  value                     # JSON
  source                    # explicit | inferred
  status                    # confirmed | candidate
  created_at, last_used_at, updated_by
```

Keys come from a versioned registry file (`prefs/registry.yaml`) defining type, allowed
values, and where each key is consumed (report builder, tone, thresholds). Unknown keys are
rejected at write time — this is what keeps the store auditable and editable rather than a
junk drawer.

### B.2 Read/write paths

- **Read:** whole confirmed profile injected at session start (it is small by
  construction; no retrieval, no embeddings). Consumed by report-builder skills via
  template variables, not free-text interpretation, wherever mechanically possible
  (chart order MUST be mechanical).
- **Explicit write:** `prefs_set` MCP tool, callable when the manager states a preference
  ("always show EUR/USD first"). Writes `confirmed` directly, `source=explicit`, and the
  agent acknowledges in-reply.
- **Inferred write:** the Phase-A reflector may emit `PREF_CANDIDATE(key, value, evidence)`
  deltas → status `candidate`. Candidates are surfaced to the manager at the top of the
  next morning report ("I noticed you always ask for X — make it default? y/n") and only
  then confirmed. Never silently applied.
- **Edit/audit:** `prefs_list` / `prefs_delete` tools + the raw table is the audit view.

### B.3 Exit criteria

Adherence and persistence evals green (6.3), isolation test green (manager A's prefs
provably never alter manager B's output), inferred-candidate loop demonstrated end-to-end.

---

## Phase C — Findings store (build third; completes phase 1)

### C.1 Graphiti backbone (temporal facts)

- Deploy Graphiti (open-source Zep engine) with Neo4j. Custom entity types:
  `Desk, Division, CurrencyPair, Instrument, OptionTrade, HedgePosition, Client,
  Counterparty, MarketEvent, NewsItem, PatternNode`.
- Seed the org/portfolio hierarchy **statically** at setup (`part_of` edges,
  Desk→Division etc.). Do not rely on community detection to rediscover the org chart.
- Episodes ingested: nightly-run summaries, investigation transcripts, news items,
  trade-lifecycle events from atom's feed (open/amend/cancel/expire — these become
  temporal facts with validity windows, e.g. option valid Jan 14 → expiry).
- Time-axis rollups: nightly job writes per-desk daily insight digests; weekly/monthly
  RAPTOR-style rollup summaries are scheduled batch jobs, ingested back as episodes.

### C.2 Insight DAG (sidecar table, NOT inside Graphiti)

Postgres table with edges referencing Graphiti fact UUIDs and atom snapshot ids. Sidecar
because invalidation semantics (C.6) need transactional control Graphiti does not expose.

```
insight:
  id, narrative             # surface story, entity-specific
  abstraction               # generalized causal statement, entity-free (validated, C.4)
  claims[]                  # [{text, epistemic: observed|inferred|world_knowledge, conf}]
  entity_tags[], entity_type_tags[], event_class
  pattern_ids[]             # links into pattern registry
  parents[]                 # {type: zep_fact|atom_snapshot|insight, ref, as_of}
  status                    # valid | flagged_stale | superseded | retracted
  depth                     # max parent depth + 1; confidence decays with depth
  created_at, session_ref, manager_id (author), shared=true
```

Insights are **shared institutional memory** across managers; S2 controls presentation only.

### C.3 Pattern registry

```
pattern: id, name, description, embedding, instance_insight_ids[], status, created_at
```

Canonicalization gate on every new pattern proposal: embed description, cosine vs existing
patterns; > threshold (start 0.80, tune on eval) → link to existing instead of minting.
Below threshold → human review queue (reuse the Phase-A CLI). Target registry size:
dozens. If it grows past ~100, thresholds are wrong — stop and re-tune.

### C.4 Write path (validation gates are the point)

`insight_writer` subagent runs at end of morning-routine insight generation and after any
adhoc investigation that reached a conclusion:
1. Schema-forced generation of the insight object (separate narrative / abstraction /
   claims / tags fields).
2. **Abstraction validator** (hard gate): regex reject if abstraction contains tickers,
   currency-pair symbols, desk names, client ids, or absolute dates (maintain deny-lists
   from atom's reference data); then one LLM check: "is this statement true of the
   general pattern, not just this instance?" Fail → one rewrite attempt → else store with
   `status=flagged_stale` equivalent (`needs_review`), never silently as valid.
3. Pattern canonicalization (C.3).
4. Parent capture: the `PostToolUse` hook has been logging every atom snapshot id and
   Graphiti fact UUID the session touched; the writer selects the subset actually load-
   bearing for each claim. No insight commits with zero parents.

### C.5 Retrieval + composition + escalation

Retrieval = union of three surfaces, merged and deduped:
(a) Graphiti native hybrid search for facts (entity + validity-window scoped);
(b) embedding search over insight **abstractions** (not narratives);
(c) pattern-node hop: situation → pattern → instance insights (PPR-style traversal,
    two-hop cap for phase 1).

Composition: single LLM pass over the full scoped set (facts + retrieved insights),
producing claims with epistemic tags.

**Stopping rule / escalation:** when the task is DoD attribution, coverage is
quantitative — if composed explanation accounts for < 90% of the move, or for
non-quantitative questions an LLM answerability check fails, spawn the `investigator`
subagent running the iterative loop: treat intermediate inferences as new retrieval
seeds, re-scope, re-compose. Depth cap 3, then return partial answer with explicit
"unattributed residual" statement. (This loop is the original entity-combination
algorithm, demoted to fallback.)

### C.6 Invalidation propagation

Triggers: (a) Graphiti closes/contradicts a fact's validity edge (native behavior on
contradictory ingestion); (b) atom restatement events (cancel/correct feed) mapped to
fact contradictions; (c) manual retraction tool.

Propagation job (transactional, runs on trigger):
1. Find insights with the invalidated ref in `parents[]` → set `flagged_stale`,
   record cause.
2. Recurse upward through insight→insight edges.
3. Decrement affected patterns' live instance counts.
4. Any ACE rule whose `justification_ptr` pattern dropped below its promotion evidence
   bar → status `candidate` again + notification into the review CLI ("evidence
   weakened: reconfirm or retire").
5. Retrieval NEVER silently drops flagged insights — they are returned with their flag
   so the agent can say "this prior conclusion was superseded because X."

### C.7 Promotion (closes the funnel)

Nightly job scans patterns crossing the bar (default: >= 2 valid instances, OR 1 instance
with severity above threshold) → drafts an ACE candidate entry with
`justification_ptr=pattern_id` → Phase-A approval flow. Promotion is the ONLY path from
findings to playbook; the reflector must not independently distill incident conclusions
(single write path rule, Section 1).

---

## 6. Evaluation

### 6.1 Harness (build once, in Phase A)

`evals/harness.py`: runs a scenario = {fixture DBs for S1/S2/S3, mock-atom MCP server
with scripted snapshots/feeds, a session script (one or many sessions), scoring fns}.
Treatment and baseline run through identical `ClaudeSDKClient` configs differing only in
mounted stores. Mock atom is an in-process MCP server (`create_sdk_mcp_server`) serving
deterministic fixtures — this is what makes every eval below runnable "inside" the SDK
rather than as detached retrieval unit tests. Metrics logged per run: task score, token
in/out, tool calls, latency, and full transcript for judge-based scoring.

Scoring: programmatic wherever possible (chart order, flag presence, retrieval hit ids);
LLM-judge (pinned model + rubric) only where unavoidable (insight quality), always with
a 30-sample human-agreement calibration before trusting it.

### 6.2 Phase A suite — what production needs: "does behavior improve and persist?"

Primary (custom) — `evals/ace/`:
- **Adherence:** 30 synthetic morning-routine + drilldown tasks on mock atom; playbook
  seeded with 15 known rules (incl. the orphaned-hedge check). Metric: % rules correctly
  fired when triggered, % false fires. Baseline gets the same rules pasted once in a
  prior-session transcript file — measures injection vs "it's somewhere in history."
- **Learning stream:** 40-task stream where feedback is given after failures
  (StreamBench-style protocol, our tasks). Metric: success-rate slope across the stream;
  treatment (reflector on) vs baseline (no reflection). This is the production question —
  "does the manager correct the agent once, or forever?"
- **Scope isolation:** manager-scoped rule must fire for A, never for B.
- **Budget stress:** 300 junk candidates injected; verify pruning holds the 2k budget and
  adherence on the 15 real rules does not degrade.

Public benchmarks: none adopted as primary. Justification: ACE's own paper evaluates on
AppWorld and FiNER — AppWorld measures general agentic competence (confounds SDK quality
with playbook quality) and FiNER measures token-labeling accuracy, not procedure
retention across sessions. LifelongAgentBench/StreamBench domains (DB ops, coding) don't
transfer, but we adopt StreamBench's *protocol* (streamed tasks + feedback + slope
metric) as stated above. Measuring the actual production behavior on our task
distribution beats a mismatched public number.

### 6.3 Phase B suite — "are stated preferences honored, forever, for the right person?"

Primary (custom) — `evals/prefs/`:
- **Adherence:** seeded profiles → report tasks where prefs must alter output; chart
  order and threshold flags scored programmatically (this is why B.2 mandates mechanical
  consumption), tone prefs by judge.
- **Persistence:** pref stated conversationally in session 1 (`prefs_set` path), checked
  in sessions 2, 5, 10 via harness multi-session runs. Baseline: transcript-grep memory.
- **Isolation & audit:** cross-manager bleed test; delete-then-verify test (deleted pref
  must stop applying next session — auditability includes revocation).
- **Inference loop:** scripted manager who repeats a request 3 sessions running; candidate
  must appear, must NOT auto-apply before confirmation, must apply after.

Public benchmark: **PrefEval (protocol + metrics, our domain data).** Justification: it is
the one benchmark measuring exactly our production property — unprompted preference
adherence over long horizons, including violation types (ignore, contradict,
hallucinate). Its generic consumer topics (restaurants, travel) make its *data* useless
for us, so we port its evaluation protocol onto risk-report tasks. Running its stock
dataset would measure nothing about chart ordering — rejected as a vanity number.

### 6.4 Phase C suite — "recall the right precedent, respect time, retract cleanly"

Primary (custom) — `evals/findings/` built on a **synthetic incident corpus generator**
(`evals/findings/gen.py`): produces N months of mock-atom feeds + news with planted
structure — cross-incident analog pairs (same pattern, different pair/desk/months apart,
e.g. orphaned-hedge in USD/JPY month 1 and EUR/USD month 4), scheduled events,
restatements, and distractor incidents that are textually similar but causally different
(the trap flat retrieval falls into). Ground truth is emitted by the generator.
- **Cross-incident recall:** "have we seen this before?" tasks. Metrics: analog hit-rate
  @k, distractor rejection rate. Baseline: transcript-grep. This is THE metric the DAG
  exists for.
- **Temporal QA:** period-over-period and "when did X change / what was believed as-of D"
  questions scored against generator ground truth. Tests Graphiti validity windows +
  as-of retrieval.
- **Invalidation correctness:** inject restatements mid-corpus; re-ask previously
  answered questions. Metrics: stale-answer rate (agent asserts retracted conclusion
  without flag = fail), over-invalidation rate (untouched insights wrongly flagged),
  ACE-rule surfacing when justification weakens (C.6 step 4).
- **Amortization:** repeat/adjacent queries across sessions; metric: token cost and
  latency of query k vs query 1 at equal answer quality.
- **Write-path quality:** abstraction validator pass rate, pattern canonicalization
  precision/recall against generator's known pattern labels.

Public benchmark: **LongMemEval (S)** as the external sanity check, wiring S3 as the
memory backend over its ~50-session histories and letting the SDK agent answer through
our retrieval tools. Justification: (a) recent analyses found LoCoMo largely insensitive
to memory architecture — good scores there don't discriminate, so **LoCoMo is explicitly
rejected**; (b) LongMemEval's question types — temporal reasoning, knowledge updates,
multi-session synthesis, and *abstention* (knowing it doesn't know) — are one-to-one
with production behaviors we care about (as-of correctness, restatements, cross-incident
synthesis, not hallucinating precedents). Expect no lift on its single-session-recall
slice (that's not what S3 adds); report per-question-type, and treat temporal +
knowledge-update + abstention slices as the pass/fail signal. HotpotQA/MuSiQue-class
multi-hop sets are rejected as primary: single-corpus hop-finding, no longitudinal
sessions, no invalidation — and their prior role here died with the predictor.

### 6.5 Ship gates (phase 1 done when)

- A: adherence uplift and positive learning slope vs baseline; isolation clean.
- B: 100% programmatic adherence on mechanical prefs; persistence at session 10;
  zero bleed; revocation honored.
- C: analog hit-rate materially above transcript-grep baseline (target: 2x, tune after
  first run); stale-answer rate < 5%; over-invalidation < 5%; LongMemEval temporal +
  update + abstention slices >= baseline agent with full-history-in-context where it
  fits (honesty check: if the whole history fits in context, memory must at least match
  it while costing fewer tokens).

---

## 7. Suggested repo layout

```
risk-agent-memory/
  agent/            # ClaudeAgentOptions builders, subagent defs, hooks
  stores/ace/       # models, injection compiler, reflector prompts, review CLI
  stores/prefs/     # models, registry.yaml, MCP tools
  stores/findings/  # graphiti setup, dag models, writer, retrieval, invalidation
  mock_atom/        # fixture MCP server + feed generator
  evals/            # harness.py, ace/, prefs/, findings/ (incl. gen.py), longmemeval/
  baseline/         # frozen baseline agent config
```

## 8. Implementation notes for Claude Code

- Freeze `baseline/` and the harness before Phase A store code; every later PR runs evals.
- Pin the judge model and all thresholds (dedup 0.85, pattern 0.80, coverage 0.90) in one
  `config.py`; they are tuning knobs, not constants.
- Verify current `claude-agent-sdk` hook names/signatures against the official docs at
  build time (they have changed across versions; e.g. confirm `SessionEnd` vs `Stop`
  availability in the pinned version).
- Graphiti requires an LLM + embedder config for ingestion; route through the same
  Anthropic account and log token spend — ingestion cost is part of the C.4 budget story.
