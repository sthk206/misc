"""
Central paths / DB config for the poc_eval harness.

Everything is overridable via environment variables so the same code runs against
a local brew MySQL (default) or any other MySQL-compatible endpoint. The data
layout deliberately mirrors the repo: per-table `.xlsx` files (the T / table
modality) live in an `excel/` dir, per-document key/value JSON files (the D /
document-text modality) live in a `doc/` dir, and generated schema JSON (consumed
by the NL2SQL step, exactly like offline `data_persistent.py`) lives in `schema/`.
"""
from __future__ import annotations

import os

# --- repo locations -------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
SOURCE_PDF = os.environ.get("POC_SOURCE_PDF", os.path.join(REPO_ROOT, "corp-10k-2025.pdf"))
BENCHMARK_FILE = os.path.join(HERE, "benchmark", "benchmark_questions.json")

# --- generated-data root --------------------------------------------------------------
# Each parser "version" gets its own self-contained dataset dir so the three systems
# never cross-contaminate. version is one of: "auto", "perfect".
DATA_ROOT = os.environ.get("POC_DATA_ROOT", os.path.join(HERE, "data"))


def dataset_dir(version: str) -> str:
    return os.path.join(DATA_ROOT, version)


def excel_dir(version: str) -> str:
    return os.path.join(dataset_dir(version), "excel")


def doc_dir(version: str) -> str:
    return os.path.join(dataset_dir(version), "doc")


def schema_dir(version: str) -> str:
    return os.path.join(dataset_dir(version), "schema")


def embedding_path(version: str, system: str) -> str:
    return os.path.join(dataset_dir(version), f"embedding_{system}.pkl")


# --- MySQL ----------------------------------------------------------------------------
# Defaults match a fresh `brew install mysql` (root, no password, local server).
# Each parser version gets its own database so auto/perfect tables stay isolated.
DB_HOST = os.environ.get("POC_DB_HOST", "127.0.0.1")
DB_PORT = int(os.environ.get("POC_DB_PORT", "3306"))
DB_USER = os.environ.get("POC_DB_USER", "root")
DB_PASSWORD = os.environ.get("POC_DB_PASSWORD", "")
DB_CHARSET = os.environ.get("POC_DB_CHARSET", "utf8mb4")
DB_PREFIX = os.environ.get("POC_DB_PREFIX", "tablerag_poc")


def database_name(version: str) -> str:
    return f"{DB_PREFIX}_{version}"


def server_url() -> str:
    """SQLAlchemy URL to the server (no specific database) -- used to CREATE DATABASE."""
    return (
        f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/"
        f"?charset={DB_CHARSET}"
    )


def database_url(version: str) -> str:
    return (
        f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/"
        f"{database_name(version)}?charset={DB_CHARSET}"
    )


# --- retrieval / agent knobs (mirror the repo) ----------------------------------------
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RECALL_NUM = 30
RERANK_NUM = 5
MAX_ITER = 5
