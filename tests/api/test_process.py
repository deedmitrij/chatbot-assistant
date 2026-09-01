import pytest
from main import app


class FakeChatManager:
    """Duck-typed stand-in for ChatManager. No real LLM/vector DB/Telegram/SQLite involved."""

    def __init__(self, process_response):
        self._process_response = process_response
        self.calls = []

    def process_message(self, user_msg):
        self.calls.append(user_msg)
        return self._process_response


@pytest.fixture
def client():
    return app.test_client()


def test_process_returns_503_when_chat_manager_uninitialized(client, monkeypatch):
    """
    Matches the same 503 guard already covered for /api/check_status in
    test_check_status.py. A valid message is sent because handle_chat()
    checks the message before it checks chat_manager, so an empty/missing
    message would return 400 regardless of manager state.
    """
    monkeypatch.delattr(app, "chat_manager", raising=False)

    response = client.post("/api/process", json={"message": "Do you have a pool?"})

    assert response.status_code == 503
    assert response.get_json() == {"error": "System initializing, please try again in a moment."}


def test_process_returns_400_when_message_key_missing(client, monkeypatch):
    fake_manager = FakeChatManager({"status": "direct", "answer": "unused"})
    monkeypatch.setattr(app, "chat_manager", fake_manager, raising=False)

    response = client.post("/api/process", json={})

    assert response.status_code == 400
    assert response.get_json() == {"error": "No message provided"}
    assert fake_manager.calls == []


def test_process_returns_400_when_message_is_empty_string(client, monkeypatch):
    fake_manager = FakeChatManager({"status": "direct", "answer": "unused"})
    monkeypatch.setattr(app, "chat_manager", fake_manager, raising=False)

    response = client.post("/api/process", json={"message": ""})

    assert response.status_code == 400
    assert response.get_json() == {"error": "No message provided"}
    assert fake_manager.calls == []


def test_process_returns_direct_answer_result_verbatim(client, monkeypatch):
    """Happy path: the endpoint must serialize ChatManager.process_message's dict unchanged."""
    fake_manager = FakeChatManager({"status": "direct", "answer": "Yes, we have a pool."})
    monkeypatch.setattr(app, "chat_manager", fake_manager, raising=False)

    response = client.post("/api/process", json={"message": "Do you have a pool?"})

    assert response.status_code == 200
    assert response.get_json() == {"status": "direct", "answer": "Yes, we have a pool."}
    assert fake_manager.calls == ["Do you have a pool?"]


def test_process_returns_pending_result_verbatim(client, monkeypatch):
    """Happy path: the low-confidence/HITL branch's dict shape must also pass through unchanged."""
    fake_manager = FakeChatManager({"status": "pending", "request_id": "req-123"})
    monkeypatch.setattr(app, "chat_manager", fake_manager, raising=False)

    response = client.post("/api/process", json={"message": "Can I bring my pet elephant?"})

    assert response.status_code == 200
    assert response.get_json() == {"status": "pending", "request_id": "req-123"}
    assert fake_manager.calls == ["Can I bring my pet elephant?"]


def test_process_returns_400_instead_of_crashing_on_null_json_body(client, monkeypatch):
    """
    Regression test for: POST /api/process with a JSON body of literal `null`
    (valid JSON, parses to None) must degrade to the same graceful 400
    "No message provided" response as a missing/empty message - matching
    this endpoint's own established error contract - instead of raising an
    unhandled AttributeError ('NoneType' object has no attribute 'get') that
    surfaces to the client as a bare 500.
    """
    fake_manager = FakeChatManager({"status": "direct", "answer": "unused"})
    monkeypatch.setattr(app, "chat_manager", fake_manager, raising=False)

    response = client.post("/api/process", data="null", content_type="application/json")

    assert response.status_code == 400
    assert response.get_json() == {"error": "No message provided"}
    assert fake_manager.calls == []


def test_process_returns_400_for_non_object_json_string_body(client, monkeypatch):
    """
    Regression test for: POST /api/process with a syntactically valid JSON
    body that decodes to a non-dict, truthy value (e.g. a bare string) must
    return the same 400 "No message provided" contract as a missing message,
    instead of raising an unhandled AttributeError ('str' object has no
    attribute 'get'). `data = request.get_json(silent=True) or {}` only
    rescues *falsy* non-dict JSON (null, [], "", 0, false) - a truthy
    non-dict body like "hello" still slips through and crashes.
    """
    fake_manager = FakeChatManager({"status": "direct", "answer": "unused"})
    monkeypatch.setattr(app, "chat_manager", fake_manager, raising=False)

    response = client.post("/api/process", json="hello")

    assert response.status_code == 400
    assert response.get_json() == {"error": "No message provided"}
    assert fake_manager.calls == []


def test_process_returns_400_for_non_object_json_array_body(client, monkeypatch):
    """
    Same defect as the string case above, but for a non-empty JSON array
    (also truthy, so also not rescued by `or {}`).
    """
    fake_manager = FakeChatManager({"status": "direct", "answer": "unused"})
    monkeypatch.setattr(app, "chat_manager", fake_manager, raising=False)

    response = client.post("/api/process", json=["not", "a", "dict"])

    assert response.status_code == 400
    assert response.get_json() == {"error": "No message provided"}
    assert fake_manager.calls == []
