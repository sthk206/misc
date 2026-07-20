import pytest

from risk_agent_memory.stores.prefs.models import (
    render_candidate_prompts,
    render_profile_block,
)
from risk_agent_memory.stores.prefs.registry import InvalidPrefValue, UnknownPrefKey


def test_unknown_key_rejected(prefs):
    with pytest.raises(UnknownPrefKey):
        prefs.set("mgr_a", "layout.made_up_key", 1)


def test_value_validation(prefs):
    with pytest.raises(InvalidPrefValue):
        prefs.set("mgr_a", "tone.verbosity", "shouty")
    with pytest.raises(InvalidPrefValue):
        prefs.set("mgr_a", "thresholds.dod_flag_ccy_pair", 999)
    with pytest.raises(InvalidPrefValue):
        prefs.set("mgr_a", "layout.chart_order", ["EURUSD", "FAKEPAIR"])
    prefs.set("mgr_a", "tone.verbosity", "terse")
    assert prefs.profile("mgr_a")["tone.verbosity"] == "terse"


def test_explicit_write_is_confirmed_immediately(prefs):
    prefs.set("mgr_a", "layout.chart_order", ["EURUSD", "USDJPY"])
    assert prefs.profile("mgr_a")["layout.chart_order"] == ["EURUSD", "USDJPY"]


def test_candidate_never_silently_applied(prefs):
    prefs.propose_candidate("mgr_a", "tone.verbosity", "terse", evidence="asked 3x")
    assert prefs.profile("mgr_a") == {}                      # not applied
    assert "confirmation" in render_candidate_prompts(prefs, "mgr_a")
    prefs.confirm("mgr_a", "tone.verbosity")
    assert prefs.profile("mgr_a")["tone.verbosity"] == "terse"


def test_candidate_rejection(prefs):
    prefs.propose_candidate("mgr_a", "tone.verbosity", "detailed")
    prefs.reject_candidate("mgr_a", "tone.verbosity")
    with pytest.raises(KeyError):
        prefs.confirm("mgr_a", "tone.verbosity")
    assert prefs.profile("mgr_a") == {}


def test_candidate_does_not_override_confirmed(prefs):
    prefs.set("mgr_a", "tone.verbosity", "terse")
    prefs.propose_candidate("mgr_a", "tone.verbosity", "detailed")
    assert prefs.profile("mgr_a")["tone.verbosity"] == "terse"


def test_isolation_across_managers(prefs):
    prefs.set("mgr_a", "tone.verbosity", "terse")
    assert prefs.profile("mgr_b") == {}
    assert render_profile_block(prefs.profile("mgr_b"), prefs.registry) == ""


def test_delete_is_revocation(prefs):
    prefs.set("mgr_a", "thresholds.stale_hedge_days", 5)
    prefs.delete("mgr_a", "thresholds.stale_hedge_days")
    assert prefs.profile("mgr_a") == {}
    assert prefs.all_rows("mgr_a") == []      # audit view reflects deletion


def test_profile_block_labels_consumer(prefs):
    prefs.set("mgr_a", "layout.chart_order", ["EURUSD"])
    block = render_profile_block(prefs.profile("mgr_a"), prefs.registry)
    assert "layout.chart_order" in block
    assert "report_builder" in block
