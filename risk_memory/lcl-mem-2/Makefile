PY := .venv/bin/python

.PHONY: install test test-fast eval-findings review

install:
	uv pip install -e '.[dev]'

test:
	$(PY) -m pytest -q

test-fast:
	$(PY) -m pytest -q -m "not integration"

eval-findings:
	.venv/bin/ram eval findings

review:
	.venv/bin/ram playbook review
