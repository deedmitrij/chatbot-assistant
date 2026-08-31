import pytest
from main import app


class FakeChatManager:
    """Duck-typed stand-in for ChatManager. No real LLM/vector DB/Telegram/SQLite involved."""

    def __init__(self, status_response):
        self._status_response = status_response
        self.calls = []

    def check_status(self, req_id):
        self.calls.append(req_id)
        return self._status_response


@pytest.fixture
def client():
    return app.test_client()


def test_check_status_returns_503_when_chat_manager_uninitialized(client, monkeypatch):
    """
    Regression test for: /api/check_status/<req_id> must degrade gracefully
    (503) when app.chat_manager hasn't been set yet, matching the guard already
    used by /api/process and /api/operator/call, instead of raising an
    unhandled AttributeError.

    We explicitly delete app.chat_manager here (rather than relying on it never
    having been set) so this test cannot pass by accident due to import order
    or state left behind by another test/module that already called
    bootstrap_manager() or otherwise set the attribute on the shared Flask app.
    """
    monkeypatch.delattr(app, "chat_manager", raising=False)

    response = client.get("/api/check_status/some-request-id")

    assert response.status_code == 503
    assert response.get_json() == {"error": "System initializing, please try again in a moment."}


def test_check_status_returns_data_when_chat_manager_initialized(client, monkeypatch):
    """Happy-path check: existing behavior must be unchanged once chat_manager is set."""
    fake_manager = FakeChatManager({"status": "completed", "answer": "Yes, we have a pool."})
    monkeypatch.setattr(app, "chat_manager", fake_manager, raising=False)

    response = client.get("/api/check_status/abc-123")

    assert response.status_code == 200
    assert response.get_json() == {"status": "completed", "answer": "Yes, we have a pool."}
    assert fake_manager.calls == ["abc-123"]
