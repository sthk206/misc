from __future__ import annotations

import json
import re

import pytest

from lcm_mem.encoder.embed import HashingEmbedder
from lcm_mem.llm.gateway import CachedGateway


class FakeChoice:
    def __init__(self, text: str):
        self.message = type("M", (), {"content": text})()


class FakeUsage:
    prompt_tokens = 10
    completion_tokens = 5


class FakeResponse:
    def __init__(self, text: str):
        self.choices = [FakeChoice(text)]
        self.usage = FakeUsage()


class FakeClient:
    """OpenAI-style client with scripted behavior. `handler` maps the last user
    message to a response string."""

    def __init__(self, handler):
        self.calls = 0
        outer = self

        class _Completions:
            def create(self, model, messages, temperature=0.0, **kw):
                outer.calls += 1
                return FakeResponse(handler(messages[-1]["content"]))

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()


def default_handler(content: str) -> str:
    """Reasonable scripted responses for each prompt family used in tests."""
    if content.startswith("Rewrite the question"):
        q = re.search(r"Question: (.*)\n", content).group(1)
        a = re.search(r"Answer: (.*)\n", content).group(1)
        return f"{q.rstrip('?')} is {a}."
    if content.startswith("Combine the two facts"):
        fa = re.search(r"Fact 1: (.*)\n", content).group(1)
        fb = re.search(r"Fact 2: (.*)\n", content).group(1)
        return f"{fa.rstrip('.')} and {fb}"
    if content.startswith("Extract the atomic"):
        return json.dumps([
            {"fact": "Alice works at Acme.", "entities": ["Alice", "Acme"], "salience": 7},
            {"fact": "Acme is based in Paris.", "entities": ["Acme", "Paris"], "salience": 5},
        ])
    if content.startswith("For each numbered pair"):
        n = len(re.findall(r"^\d+\. OLD:", content, flags=re.M))
        return json.dumps(["unrelated"] * n)
    if content.startswith("Given fact A and fact B"):
        return json.dumps({"inference": "NONE", "confidence": 0.0,
                           "used_world_knowledge": False})
    if content.startswith("Can the question be answered"):
        return "no"
    if content.startswith("List the named entities"):
        return json.dumps([])
    if content.startswith("Answer the question"):
        return "I don't know"
    if content.startswith("How useful"):
        return "5"
    return "OK"


@pytest.fixture
def fake_gateway(tmp_path):
    client = FakeClient(default_handler)
    gw = CachedGateway(client=client, cache_path=tmp_path / "cache.sqlite")
    gw._test_client = client  # expose for call-count assertions
    return gw


@pytest.fixture
def hash_embedder():
    return HashingEmbedder(dim=32)
