"""
LLM + embedding gateway shim.

Both systems under evaluation (baseline RAG and TableRAG) call ONLY through this
module, so they are guaranteed to use the identical chat model, embedding model,
temperature, and endpoint -- which is what makes the comparison fair.

The actual credentials are intentionally left as a placeholder. The user runs this
later behind their own OpenAI-compatible "LLM gateway": they have a function that
returns a bearer token, which is passed straight into `OpenAI(api_key=...)`.

>>> TO WIRE THIS UP, do exactly two things: <<<
  1. Implement `get_bearer_token()` below (drop in your existing token function).
  2. Set GATEWAY_URL / CHAT_MODEL / EMBED_MODEL to your gateway's real values
     (or export them as env vars POC_GATEWAY_URL / POC_CHAT_MODEL / POC_EMBED_MODEL).

Nothing else in the codebase needs to change.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from openai import OpenAI

# --------------------------------------------------------------------------------------
# Configuration (placeholders -- override via env vars or edit directly).
# --------------------------------------------------------------------------------------
GATEWAY_URL: str = os.environ.get("POC_GATEWAY_URL", "https://your-llm-gateway.example.com/v1")
CHAT_MODEL: str = os.environ.get("POC_CHAT_MODEL", "PLACEHOLDER_CHAT_MODEL")
EMBED_MODEL: str = os.environ.get("POC_EMBED_MODEL", "PLACEHOLDER_EMBED_MODEL")

# Mirrors the repo's online_inference/chat_utils.py (temperature=0.1) for fidelity.
TEMPERATURE: float = 0.1

# Embedding batch size kept modest to stay within typical gateway request limits.
EMBED_BATCH_SIZE: int = 64


def get_bearer_token() -> str:
    """Return a bearer token for the LLM gateway.

    >>> REPLACE THE BODY with your own token function, e.g.:

        from my_company.auth import fetch_gateway_token
        return fetch_gateway_token()

    The returned string is passed directly as `OpenAI(api_key=<token>)`.
    """
    token = os.environ.get("POC_GATEWAY_TOKEN")
    if token:
        return token
    raise NotImplementedError(
        "llm_gateway.get_bearer_token() is a placeholder. Implement it with your "
        "token function, or export POC_GATEWAY_TOKEN for a quick test run."
    )


def get_client() -> OpenAI:
    """Build an OpenAI-compatible client pointed at the gateway.

    A fresh client is built per call so short-lived bearer tokens stay valid;
    client construction is cheap.
    """
    return OpenAI(api_key=get_bearer_token(), base_url=GATEWAY_URL)


def chat(
    messages: list[dict[str, Any]],
    tools: Optional[list[dict[str, Any]]] = None,
    tool_choice: Optional[Any] = None,
    temperature: float = TEMPERATURE,
    model: str = CHAT_MODEL,
):
    """Chat completion. Returns the raw `message` object from the first choice
    (so callers can read `.content` and `.tool_calls`), mirroring the repo's
    `get_chat_result`."""
    client = get_client()
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if tools is not None:
        kwargs["tools"] = tools
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice
    completion = client.chat.completions.create(**kwargs)
    return completion.choices[0].message


def chat_text(messages: list[dict[str, Any]], **kwargs) -> str:
    """Convenience wrapper returning just the assistant text content."""
    msg = chat(messages, **kwargs)
    return (msg.content or "").strip()


def embed(texts: list[str], model: str = EMBED_MODEL) -> list[list[float]]:
    """Embed a list of texts via `client.embeddings.create(model, input=...)`.

    Returns one vector per input text, order preserved. Batched to respect
    gateway request-size limits.
    """
    client = get_client()
    vectors: list[list[float]] = []
    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[start : start + EMBED_BATCH_SIZE]
        resp = client.embeddings.create(model=model, input=batch)
        # Sort by index defensively; the API preserves order but this is cheap insurance.
        for item in sorted(resp.data, key=lambda d: d.index):
            vectors.append(item.embedding)
    return vectors


def config_banner() -> str:
    """One-line summary of the active config, printed at run start so the user can
    confirm both systems share identical settings."""
    return (
        f"[gateway] url={GATEWAY_URL} chat_model={CHAT_MODEL} "
        f"embed_model={EMBED_MODEL} temperature={TEMPERATURE}"
    )
