"""
Re-export the repo's prompt strings verbatim by loading them straight from the repo
source files. This guarantees the harness uses byte-identical prompts to TableRAG --
"keep everything else the same to the repo" -- and that the mock_gateway's substring
routing (which keys off these exact prompts) keeps working.
"""
from __future__ import annotations

import importlib.util
import os

from poc_eval import config


def _load(rel_path: str, mod_name: str):
    path = os.path.join(config.REPO_ROOT, rel_path)
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_online = _load(os.path.join("online_inference", "prompt.py"), "repo_online_prompt")
_offline = _load(
    os.path.join("offline_data_ingestion_and_query_interface", "src", "prompt.py"),
    "repo_offline_prompt",
)

SYSTEM_EXPLORE_PROMPT = _online.SYSTEM_EXPLORE_PROMPT
COMBINE_PROMPT = _online.COMBINE_PROMPT
EVALUATION_PRONPT = _online.EVALUATION_PRONPT
NL2SQL_SYSTEM_PROMPT = _offline.NL2SQL_SYSTEM_PROMPT
NL2SQL_USER_PROMPT = _offline.NL2SQL_USER_PROMPT
