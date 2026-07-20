import pytest

from lcm_mem.llm.gateway import CachedGateway, CacheMiss, parse_json_response
from tests.conftest import FakeClient


def test_cache_hit_avoids_second_call(tmp_path):
    client = FakeClient(lambda c: "hello")
    gw = CachedGateway(client=client, cache_path=tmp_path / "c.sqlite")
    msgs = [{"role": "user", "content": "hi"}]
    assert gw.chat(msgs, model="m") == "hello"
    assert gw.chat(msgs, model="m") == "hello"
    assert client.calls == 1
    assert gw.usage.calls == 1
    assert gw.usage.cache_hits == 1


def test_cache_key_sensitivity(tmp_path):
    client = FakeClient(lambda c: "x")
    gw = CachedGateway(client=client, cache_path=tmp_path / "c.sqlite")
    msgs = [{"role": "user", "content": "hi"}]
    gw.chat(msgs, model="m1")
    gw.chat(msgs, model="m2")                  # different model -> new call
    gw.chat(msgs, model="m1", temperature=0.5)  # different temp -> new call
    assert client.calls == 3


def test_cache_survives_reopen(tmp_path):
    path = tmp_path / "c.sqlite"
    client = FakeClient(lambda c: "persisted")
    CachedGateway(client=client, cache_path=path).chat(
        [{"role": "user", "content": "q"}], model="m"
    )
    gw2 = CachedGateway(client=FakeClient(lambda c: "SHOULD NOT BE CALLED"),
                        cache_path=path)
    assert gw2.chat([{"role": "user", "content": "q"}], model="m") == "persisted"


def test_dry_run_raises_on_miss(tmp_path):
    gw = CachedGateway(client=None, cache_path=tmp_path / "c.sqlite", dry_run=True)
    with pytest.raises(CacheMiss):
        gw.chat([{"role": "user", "content": "new"}], model="m")


def test_usage_counter(tmp_path):
    client = FakeClient(lambda c: "y")
    gw = CachedGateway(client=client, cache_path=tmp_path / "c.sqlite")
    gw.chat([{"role": "user", "content": "a"}], model="m")
    gw.chat([{"role": "user", "content": "b"}], model="m")
    assert gw.usage.prompt_tokens == 20
    assert gw.usage.completion_tokens == 10
    assert gw.usage.by_model == {"m": 2}


def test_parse_json_response_fenced():
    assert parse_json_response('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_response('noise before {"a": [1, 2]} noise after') == {"a": [1, 2]}
    assert parse_json_response('["x", "y"]') == ["x", "y"]
