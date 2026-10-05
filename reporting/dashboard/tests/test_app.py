import json

from reporting.dashboard.app import create_app


def _client(tmp_path, latest_content=None):
    latest = tmp_path / "latest.json"
    if latest_content is not None:
        latest.write_text(latest_content, encoding="utf-8")
    return create_app(latest, tmp_path / "events.jsonl").test_client()


def test_index_serves_dashboard_page(tmp_path):
    response = _client(tmp_path).get("/")
    assert response.status_code == 200
    assert b"Evaluation Dashboard" in response.data


def test_results_with_no_reports_is_empty_not_error(tmp_path):
    response = _client(tmp_path).get("/api/results")
    assert response.status_code == 200
    body = response.get_json()
    assert body["rows"] == []
    assert body["source"] is None
    assert body["summary"]["overall"]["total"] == 0


def test_results_malformed_latest_json_returns_error(tmp_path):
    response = _client(tmp_path, "{broken").get("/api/results")
    assert response.status_code == 500
    assert "latest.json" in response.get_json()["error"]


def test_results_unknown_scope_is_rejected(tmp_path):
    response = _client(tmp_path, json.dumps({"results": []})).get("/api/results?scope=../../etc")
    assert response.status_code == 400
