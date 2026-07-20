PY := .venv/bin/python
LCM := .venv/bin/lcm

.PHONY: install test data phase1 audit lme

install:
	uv pip install -e '.[dev,faiss]'

test:
	$(PY) -m pytest -q

# Phase 0 deliverable: musique_triples.{train,val,test}.jsonl + cached embeddings
data:
	$(LCM) build-triples configs/data.yaml

audit:
	$(LCM) audit-triples configs/data.yaml --n 50

phase1:
	$(LCM) phase1 configs/phase1.yaml

lme:
	$(LCM) longmemeval configs/longmemeval.yaml
