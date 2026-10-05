"""Standalone local Flask app for the Evaluation Dashboard.

Run from the project root:
    python -m reporting.dashboard.app
then open http://127.0.0.1:5050/

Kept separate from main.py so viewing results needs no Chroma / LLM /
Telegram services, and evaluation data is never exposed by the chat app.
Report paths are fixed (reporting.paths); nothing is read from user input.
"""
import logging
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from reporting.dashboard.data import SCOPE_CURRENT, SCOPES, ReportLoadError, build_dashboard
from reporting.paths import EVENTS_PATH, LATEST_PATH

STATIC_DIR = Path(__file__).resolve().parent / "static"
logger = logging.getLogger(__name__)


def create_app(latest_path: Path = LATEST_PATH, events_path: Path = EVENTS_PATH) -> Flask:
    app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")

    @app.route("/")
    def index():
        return send_from_directory(STATIC_DIR, "index.html")

    @app.route("/api/results")
    def results():
        scope = request.args.get("scope", SCOPE_CURRENT)
        if scope not in SCOPES:
            return jsonify({"error": f"Unknown scope {scope!r}; expected one of {list(SCOPES)}"}), 400
        try:
            return jsonify(build_dashboard(latest_path, events_path, scope))
        except ReportLoadError as exc:
            logger.error("Evaluation report could not be loaded: %s", exc)
            return jsonify({"error": str(exc)}), 500

    return app


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    create_app().run(host="127.0.0.1", port=5050, debug=False)
