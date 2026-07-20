# risk-agent-memory

Implementation of `../implementation_v1.md` (the current plan — the v1/v2
filenames are swapped relative to their drafting order): a memory system for a
risk intelligence agent on the Claude Agent SDK.

```
atom (trading engine)   -> live MCP tools (mocked here). NEVER memorized.
S1 ACE playbook store   -> procedural memory, compiled into context each session.
S2 Preference store     -> per-manager structured profile, injected whole.
S3 Findings store       -> temporal facts + insight DAG + pattern registry.

ACE rule -> pattern -> insights (DAG) -> {facts, atom snapshots, insights}
(edges point down to evidence; invalidation propagates up)
```

## Setup

```bash
uv venv --python 3.12        # arm64; Homebrew /usr/local python is x86_64 — avoid
uv pip install -e '.[dev]'
make test                    # 47 tests, offline (hash embedder) + e5 integration
```

`claude-agent-sdk` 0.2.x is pinned by install; hook names verified against it
(`Stop`/`PreCompact`/`PostToolUse` — there is no `SessionEnd`, the spec's
fallback applies). Live runs use this machine's Claude Code auth.

## What is where

```
src/risk_agent_memory/
  config.py                 ALL pinned thresholds (dedup .85, pattern .80, coverage .90,
                            2k playbook budget, promotion bar) — tuning knobs, one place
  stores/ace/               models (delta queue, dedup gate), injection compiler,
                            reflector contract, review CLI  (Phase A)
  stores/prefs/             registry-validated store, MCP tools, candidate loop (Phase B)
  stores/findings/          temporal-fact backend (in-memory + Graphiti stub), insight
                            DAG sidecar, abstraction-validator write gates, 3-surface
                            retrieval, invalidation propagation, promotion (Phase C)
  agent/                    ClaudeAgentOptions builders, subagent defs, provenance hooks
  baseline/                 frozen stock-SDK baseline (transcript-grep memory only)
  mock_atom/                deterministic world + in-process MCP server
  evals/                    harness (treatment vs baseline), synthetic incident
                            generator with planted analogs/distractors/restatements,
                            ace / prefs / findings suites
```

## Commands

```bash
ram playbook review          # human approval queue (nothing activates without it)
ram playbook list|notifications
ram prefs list [manager] | prefs delete <manager> <key>
ram findings patterns | findings promote      # C.7 nightly promotion scan
ram eval findings            # OFFLINE suite: analog recall, distractor rejection,
                             # temporal as-of QA, invalidation correctness (no LLM)
ram eval ace --scenario adherence|learning_stream|scope_isolation [--baseline]
ram eval prefs --scenario adherence|persistence|isolation|inference [--baseline]
```

Live eval runs cost real tokens (the smoke session was ~$0.15); run baseline and
treatment arms of the same scenario and compare the emitted `results/*.json`.

## Design decisions & deviations from the spec

- **SQLite everywhere** for phase 1 (spec allows "start SQLite"); the insight-DAG
  schema is Postgres-portable. Graphiti/Neo4j is behind
  `stores/findings/graphiti_backend.py` (+ `pip install -e '.[graphiti]'`, needs a
  running Neo4j); all DAG/invalidation logic depends only on fact UUIDs, so the
  backend swap does not touch it. The in-memory backend reproduces the temporal
  semantics the evals need (validity windows, closing on contradiction).
- **Retrieval ranking**: pattern-hop reachability adds relevance mass to the
  combined ranking (C.5 "PPR-style"). This is load-bearing: the planted
  benign-unwind distractor beats flat embedding search (negation blindness) but
  loses once pattern mass is counted — the offline suite asserts exactly this.
- **Reflection runs post-session** from the Stop-hook queue (harness drains it)
  rather than inside the session, keeping reflector tokens out of the main context.

## Status

- [x] Phases A, B, C store logic complete and tested (47 tests green).
- [x] Harness + baseline + mock atom; live SDK session verified end-to-end
      (playbook rules demonstrably fired; hooks logged provenance).
- [x] Offline findings suite green: analog hit 1.0, distractor rejection 1.0,
      stale-answer 0, over-invalidation 0, as-of QA 1.0.
- [ ] Full live eval runs (6.2/6.3 adherence, learning-stream slope, PrefEval-style
      persistence) — `ram eval ...`, budget-gated.
- [ ] Graphiti/Neo4j deployment + static org seeding for production S3.
- [ ] LongMemEval-S external sanity check wiring (6.4).
