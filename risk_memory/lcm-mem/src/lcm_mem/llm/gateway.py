"""Cached adapter around the corporate LLM gateway.

The underlying client is any object exposing an OpenAI-style
``client.chat.completions.create(model=..., messages=..., temperature=...)``.
This module adds:
  (a) a SQLite disk cache keyed by SHA256 of (model, messages, temperature, tools),
  (b) retry with exponential backoff,
  (c) a token/cost counter for per-experiment logging,
  (d) a ``dry_run`` mode that returns cache-only results.

Gateway credentials/config come from the environment, never the repo.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class GatewayError(RuntimeError):
    pass


class CacheMiss(GatewayError):
    """Raised in dry_run mode when a request is not in the cache."""


@dataclass
class GatewayUsage:
    calls: int = 0
    cache_hits: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    errors: int = 0
    by_model: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "cache_hits": self.cache_hits,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "errors": self.errors,
            "by_model": dict(self.by_model),
        }


def get_default_client() -> Any:
    """Construct the corporate gateway client from environment config.

    Set ``LCM_GATEWAY_FACTORY="pkg.module:callable"`` to override the default
    ``company_gateway.LLMGateway`` import. The callable receives the parsed
    JSON from ``LCM_GATEWAY_CONFIG`` (default ``{}``).
    """
    config = json.loads(os.environ.get("LCM_GATEWAY_CONFIG", "{}"))
    factory_spec = os.environ.get("LCM_GATEWAY_FACTORY")
    if factory_spec:
        mod_name, _, attr = factory_spec.partition(":")
        import importlib

        factory = getattr(importlib.import_module(mod_name), attr)
        return factory(config)
    try:
        from company_gateway import LLMGateway  # type: ignore[import-not-found]
    except ImportError as e:
        raise GatewayError(
            "No LLM client available: `company_gateway` is not importable and "
            "LCM_GATEWAY_FACTORY is not set. Either run on the corporate "
            "environment, set LCM_GATEWAY_FACTORY to a client factory, or use "
            "dry_run=True against a warm cache."
        ) from e
    return LLMGateway(config)


def _cache_key(
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    tools: Any = None,
) -> str:
    blob = json.dumps(
        {"model": model, "messages": messages, "temperature": temperature, "tools": tools},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode()).hexdigest()


class CachedGateway:
    def __init__(
        self,
        client: Any = None,
        cache_path: str | Path = "cache/llm_cache.sqlite",
        dry_run: bool = False,
        max_retries: int = 5,
        base_delay: float = 1.0,
        max_calls: int | None = None,
    ):
        self._client = client
        self.dry_run = dry_run
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_calls = max_calls  # per-experiment budget cap; None = unlimited
        self.usage = GatewayUsage()
        self._lock = threading.Lock()
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(cache_path), check_same_thread=False)
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS llm_cache (
                   key TEXT PRIMARY KEY,
                   response TEXT NOT NULL,
                   model TEXT,
                   prompt_tokens INT,
                   completion_tokens INT,
                   created_at REAL
               )"""
        )
        self._db.commit()

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = get_default_client()
        return self._client

    def _cache_get(self, key: str) -> str | None:
        with self._lock:
            row = self._db.execute(
                "SELECT response FROM llm_cache WHERE key = ?", (key,)
            ).fetchone()
        return row[0] if row else None

    def _cache_put(
        self, key: str, response: str, model: str, pt: int, ct: int
    ) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO llm_cache VALUES (?, ?, ?, ?, ?, ?)",
                (key, response, model, pt, ct, time.time()),
            )
            self._db.commit()

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.0,
        tools: Any = None,
    ) -> str:
        key = _cache_key(model, messages, temperature, tools)
        cached = self._cache_get(key)
        if cached is not None:
            self.usage.cache_hits += 1
            return cached
        if self.dry_run:
            raise CacheMiss(f"dry_run cache miss for model={model}")
        if self.max_calls is not None and self.usage.calls >= self.max_calls:
            raise GatewayError(
                f"per-experiment call budget exhausted ({self.max_calls})"
            )

        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                kwargs: dict[str, Any] = dict(
                    model=model, messages=messages, temperature=temperature
                )
                if tools is not None:
                    kwargs["tools"] = tools
                resp = self.client.chat.completions.create(**kwargs)
                text = resp.choices[0].message.content
                pt = ct = 0
                usage = getattr(resp, "usage", None)
                if usage is not None:
                    pt = getattr(usage, "prompt_tokens", 0) or 0
                    ct = getattr(usage, "completion_tokens", 0) or 0
                self.usage.calls += 1
                self.usage.prompt_tokens += pt
                self.usage.completion_tokens += ct
                self.usage.by_model[model] = self.usage.by_model.get(model, 0) + 1
                self._cache_put(key, text, model, pt, ct)
                return text
            except Exception as e:  # noqa: BLE001 — gateway raises unknown types
                last_exc = e
                self.usage.errors += 1
                time.sleep(self.base_delay * (2**attempt))
        raise GatewayError(
            f"gateway call failed after {self.max_retries} retries"
        ) from last_exc

    def chat_json(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.0,
    ) -> Any:
        """Chat call whose response is parsed as JSON (tolerates ``` fences)."""
        text = self.chat(messages, model, temperature)
        return parse_json_response(text)


def parse_json_response(text: str) -> Any:
    s = text.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s
        if s.endswith("```"):
            s = s[: -3]
        s = s.strip()
        if s.startswith("json"):
            s = s[4:].strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        # last resort: find the outermost JSON object/array
        for open_c, close_c in (("{", "}"), ("[", "]")):
            i, j = s.find(open_c), s.rfind(close_c)
            if i != -1 and j > i:
                try:
                    return json.loads(s[i : j + 1])
                except json.JSONDecodeError:
                    continue
        raise
