from unittest.mock import MagicMock
import pytest
from backend.services.llm.llm_service import LLMService


def _fake_completion(content):
    """Builds a fake object shaped like an OpenAI ChatCompletion response."""
    message = MagicMock(content=content)
    choice = MagicMock(message=message)
    return MagicMock(choices=[choice])


@pytest.fixture
def llm_service():
    """
    Real LLMService instance. Constructing OpenAI/InferenceClient makes no
    network call by itself, so this stays a pure unit test as long as the
    actual HTTP call point (chat_client.chat.completions.create) is mocked
    in each test below.
    """
    return LLMService()


def test_get_answer_returns_dict_on_success(llm_service, monkeypatch):
    """Baseline: valid JSON content still produces the existing dict contract."""
    monkeypatch.setattr(
        llm_service.chat_client.chat.completions,
        "create",
        MagicMock(return_value=_fake_completion('{"confidence": true, "answer": "We are pet friendly."}')),
    )

    result = llm_service.get_answer("Are you pet friendly?", ["We allow pets."])

    assert result == {"confidence": True, "answer": "We are pet friendly."}


def test_get_answer_returns_safe_dict_when_completion_call_raises(llm_service, monkeypatch):
    """
    Regression test: if the underlying chat completion call raises (network
    error, timeout, rate limit, etc.), get_answer must return the same
    {confidence, answer} dict contract as the success path, with
    confidence=False, instead of a bare (str, bool) tuple.
    """
    monkeypatch.setattr(
        llm_service.chat_client.chat.completions,
        "create",
        MagicMock(side_effect=ConnectionError("simulated network failure")),
    )

    result = llm_service.get_answer("Are you pet friendly?", ["We allow pets."])

    assert isinstance(result, dict), f"Expected a dict contract, got {type(result)}: {result!r}"
    assert result["confidence"] is False
    assert isinstance(result["answer"], str)
    assert "simulated network failure" in result["answer"]


def test_get_answer_returns_safe_dict_when_model_content_is_not_json(llm_service, monkeypatch):
    """
    Regression test: if the model returns content that is not valid JSON,
    json.loads raises inside get_answer's try block; the method must still
    degrade to the same safe {confidence: False, answer: ...} dict contract,
    not a tuple.
    """
    monkeypatch.setattr(
        llm_service.chat_client.chat.completions,
        "create",
        MagicMock(return_value=_fake_completion("Sorry, I can't help with that.")),
    )

    result = llm_service.get_answer("Are you pet friendly?", ["We allow pets."])

    assert isinstance(result, dict), f"Expected a dict contract, got {type(result)}: {result!r}"
    assert result["confidence"] is False
    assert isinstance(result["answer"], str)
