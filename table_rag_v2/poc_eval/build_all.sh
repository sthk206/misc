#!/usr/bin/env bash
# One-shot: build both datasets from the PDF, ingest into MySQL, then run the eval.
# Run from the repo root:  bash poc_eval/build_all.sh [--mock]
#
# Prereqs:
#   * MySQL running locally (brew install mysql && mysql.server start), root accessible
#   * deps installed:  pip install -r poc_eval/requirements.txt
#   * for a REAL run, wire poc_eval/common/llm_gateway.py (token + URL + model names)
set -euo pipefail
cd "$(dirname "$0")/.."

MOCK_FLAG="${1:-}"

echo "== build datasets (auto + perfect) =="
python -m poc_eval.parse.build_dataset --version perfect
python -m poc_eval.parse.build_dataset --version auto

echo "== ingest into MySQL =="
python -m poc_eval.ingest.mysql_ingest --version perfect
python -m poc_eval.ingest.mysql_ingest --version auto

echo "== run eval (3 systems x benchmark) =="
python -m poc_eval.run_eval ${MOCK_FLAG}
