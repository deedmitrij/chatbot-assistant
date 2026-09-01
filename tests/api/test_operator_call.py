import pytest
from main import app


class FakeTelegramService:
    """Duck-typed stand-in for TelegramService. No real Telegram/network call involved."""

    def __init__(self, send_operator_call_result=None):
        self._send_operator_call_result = send_operator_call_result
        self.calls = 0

    def send_operator_call(self):
        self.calls += 1
        return self._send_operator_call_result


class FakeChatManager:
    """
    Duck-typed stand-in for ChatManager. Only exposes tg_service, which is all
    handle_operator_call() actually touches.
    """

    def __init__(self, tg_service):
        self.tg_service = tg_service


@pytest.fixture
def client():
    return app.test_client()


def test_operator_call_returns_503_when_chat_manager_uninitialized(client, monkeypatch):
    """Matches the same 503 guard already covered for /api/check_status and /api/process."""
    monkeypatch.delattr(app, "chat_manager", raising=False)

    response = client.post("/api/operator/call", json={"message": "I need help"})

    assert response.status_code == 503
    assert response.get_json() == {"error": "System initializing, please try again in a moment."}


def test_operator_call_returns_ok_and_triggers_telegram_alert(client, monkeypatch):
    fake_tg = FakeTelegramService(send_operator_call_result=999)
    fake_manager = FakeChatManager(tg_service=fake_tg)
    monkeypatch.setattr(app, "chat_manager", fake_manager, raising=False)

    response = client.post("/api/operator/call", json={"message": "I need a human"})

    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}
    assert fake_tg.calls == 1


def test_operator_call_ignores_missing_body(client, monkeypatch):
    """
    Unlike /api/process, this endpoint never reads request.json - it only
    checks chat_manager and triggers the Telegram alert. Confirmed by direct
    inspection: a request with no body/content-type at all must still succeed.
    """
    fake_tg = FakeTelegramService()
    fake_manager = FakeChatManager(tg_service=fake_tg)
    monkeypatch.setattr(app, "chat_manager", fake_manager, raising=False)

    response = client.post("/api/operator/call")

    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}
    assert fake_tg.calls == 1


def test_operator_call_ignores_missing_message_key(client, monkeypatch):
    fake_tg = FakeTelegramService()
    fake_manager = FakeChatManager(tg_service=fake_tg)
    monkeypatch.setattr(app, "chat_manager", fake_manager, raising=False)

    response = client.post("/api/operator/call", json={})

    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}
    assert fake_tg.calls == 1


def test_operator_call_ignores_empty_message(client, monkeypatch):
    fake_tg = FakeTelegramService()
    fake_manager = FakeChatManager(tg_service=fake_tg)
    monkeypatch.setattr(app, "chat_manager", fake_manager, raising=False)

    response = client.post("/api/operator/call", json={"message": ""})

    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}
    assert fake_tg.calls == 1


def test_operator_call_ignores_null_json_body(client, monkeypatch):
    fake_tg = FakeTelegramService()
    fake_manager = FakeChatManager(tg_service=fake_tg)
    monkeypatch.setattr(app, "chat_manager", fake_manager, raising=False)

    response = client.post("/api/operator/call", data="null", content_type="application/json")

    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}
    assert fake_tg.calls == 1


def test_operator_call_ignores_non_object_json_payload(client, monkeypatch):
    fake_tg = FakeTelegramService()
    fake_manager = FakeChatManager(tg_service=fake_tg)
    monkeypatch.setattr(app, "chat_manager", fake_manager, raising=False)

    response = client.post("/api/operator/call", json=["not", "a", "dict"])

    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}
    assert fake_tg.calls == 1


def test_operator_call_response_does_not_depend_on_telegram_send_result(client, monkeypatch):
    """
    handle_operator_call() discards send_operator_call()'s return value.
    Unlike /api/process, which propagates ChatManager's result verbatim, this
    endpoint always responds {"status": "ok"} regardless of whether the
    underlying Telegram call succeeded (returned a message id) or failed
    (returned None, TelegramService's own documented failure contract).
    """
    fake_tg = FakeTelegramService(send_operator_call_result=None)
    fake_manager = FakeChatManager(tg_service=fake_tg)
    monkeypatch.setattr(app, "chat_manager", fake_manager, raising=False)

    response = client.post("/api/operator/call", json={"message": "urgent"})

    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}
    assert fake_tg.calls == 1
