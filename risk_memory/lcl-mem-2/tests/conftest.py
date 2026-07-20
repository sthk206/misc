import pytest

from risk_agent_memory.embedding import HashingEmbedder
from risk_agent_memory.stores.ace.models import AceStore
from risk_agent_memory.stores.findings.backend import InMemoryFactBackend
from risk_agent_memory.stores.findings.dag import FindingsDag
from risk_agent_memory.stores.findings.writer import AbstractionValidator
from risk_agent_memory.stores.prefs.models import PrefsStore
from risk_agent_memory.stores.prefs.registry import PrefRegistry

REGISTRY_PATH = "configs/prefs_registry.yaml"
DENYLISTS_PATH = "configs/denylists.yaml"


@pytest.fixture
def embedder():
    return HashingEmbedder(dim=32)


@pytest.fixture
def ace(embedder):
    return AceStore(":memory:", embedder)


@pytest.fixture
def prefs():
    return PrefsStore(":memory:", PrefRegistry.load(REGISTRY_PATH))


@pytest.fixture
def dag(embedder):
    return FindingsDag(":memory:", embedder)


@pytest.fixture
def backend():
    return InMemoryFactBackend()


@pytest.fixture
def validator():
    return AbstractionValidator.load(DENYLISTS_PATH)
