import pytest
from backend.constants import LLMRole
from backend.services.llm.llm_service import LLMService
from reporting.recorder import Recorder
from reporting.paths import EVENTS_PATH


@pytest.fixture(scope="session")
def assistant_llm_service():
    """The real Assistant (System Under Test), shared by Custom, RAGAS and
    DeepEval Generation -- all three need the identical Assistant-role
    LLMService."""
    return LLMService(role=LLMRole.ASSISTANT)


@pytest.fixture(scope="session")
def assistant_response_cache():
    """In-memory cache so each case's Assistant response is generated once
    per pytest session and reused by every framework/metric, keyed by
    case_id."""
    return {}


@pytest.fixture(scope="session")
def recorder():
    """One Recorder per pytest session, shared by every framework, so they
    all persist to the same events.jsonl under one run_id."""
    return Recorder(EVENTS_PATH)
