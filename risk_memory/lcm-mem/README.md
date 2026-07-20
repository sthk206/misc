# LCM-Mem: Latent Composition Memory

Research implementation of the spec in `../implementation_v2.md`:

1. **Latent composition predictor** — a small network `f(emb(a), emb(b), emb(q)) →
   predicted embedding` of the LLM-composed inference, trained JEPA-style on MuSiQue
   decompositions and used as a cheap best-first-search heuristic for deciding which
   fact pairs are worth an LLM composition call.
2. **Provenance-tracked memory store** — derived facts persist with parent pointers,
   epistemic types (`observed | derived | world_bridge`), and confidence; contradiction
   of a source fact propagates soft invalidation (`stale`) through the provenance DAG.

Phase 1 is a cheap **kill experiment** that can falsify the latent-composition
hypothesis before the full system is invested in (`results/phase1_verdict.md`).

## Setup

```bash
uv venv --python 3.12          # arm64 python; Homebrew /usr/local python is x86_64
uv pip install -e '.[dev,faiss]'
make test
```

Notes for this machine (M1 Max):
- Training uses MPS automatically (`PYTORCH_ENABLE_MPS_FALLBACK=1` is set in code).
- The vector index defaults to exact numpy search; `LCM_USE_FAISS=1` opts into
  faiss-cpu, whose bundled OpenMP aborts on macOS when torch is loaded in the same
  process — leave it off here.

## LLM gateway

All generative calls go through `lcm_mem.llm.gateway.CachedGateway`, which wraps the
corporate gateway (OpenAI-style interface) with a SQLite response cache keyed by
SHA256(model, messages, temperature, tools), retry with backoff, token/cost counters,
and a `dry_run` cache-only mode. Configure via environment, never the repo:

- `LCM_GATEWAY_CONFIG` — JSON config passed to the client constructor.
- `LCM_GATEWAY_FACTORY` — optional `pkg.module:callable` overriding the default
  `company_gateway.LLMGateway` import (useful off the corporate environment).

Embeddings are always computed locally with sentence-transformers (default
`intfloat/e5-base-v2`, correct `passage:`/`query:` prefixes, per-text `.npy` cache).

## Workflow

```bash
make data      # Phase 0: MuSiQue -> musique_triples.{train,val,test}.jsonl (needs gateway)
make audit     # print 50 random triples for the manual rewrite-quality audit
make phase1    # Phase 1: train all predictors, run kill experiment, write verdict
make lme       # Phase 5: LongMemEval-S end-to-end (needs gateway + longmemeval_s.json)
lcm cache-stats
```

Every experiment is a CLI command taking one YAML config (`configs/`); every run
writes a results JSON with git hash, config, metrics, and token spend to `results/`.

## Layout

```
src/lcm_mem/
  llm/gateway.py        cached gateway adapter        llm/prompts.py   versioned prompts
  data/musique.py       triples + leakage-safe split  data/hard_negatives.py  generators
  data/longmemeval.py   benchmark loader
  encoder/embed.py      MPS embedding w/ cache        encoder/finetune.py     Phase 2
  predictor/model.py    MeanPool/LearnedPool/MLP/Attn predictor/train.py      losses+loop
  predictor/baselines.py cross-encoder frontier
  memory/store.py       SQLite facts + vector index   memory/provenance.py    invalidation
  memory/ingest.py      extraction + canonicalization memory/compose.py       best-first loop
  evals/kill_experiment.py  Phase 1 metrics + gates   evals/run_phase1.py     orchestration
  evals/longmemeval_runner.py                         evals/ablations.py
tests/                  incl. property-based invalidation tests (hypothesis)
```

## Status / next steps

- [x] Phase 0 scaffolding: gateway adapter, embedding cache, MuSiQue parsing +
      leakage-safe splitting, hard-negative generators — all unit-tested offline.
- [x] Phase 1 machinery: predictors, both losses, retrieval/tercile/discrimination/
      pair-pruning evals, verdict gates (verified end-to-end with real e5 + MPS).
- [x] Phase 3 memory store + provenance invalidation (property-tested).
- [x] Phase 4 composition loop with config-switchable pair scorers.
- [ ] Run `make data` on the corporate environment (needs the gateway), audit 50
      triples, then `make phase1` for the real verdict.
- [ ] Phase 2 encoder fine-tuning (only if Phase 1 ≥ weak pass) — `encoder/finetune.py`
      is ready; MTEB regression guard still to wire up.
- [ ] LongMemEval-S full run + ablation grid (Phase 5).
