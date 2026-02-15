from flask import Flask, send_from_directory, jsonify, request
from backend.api import routes as chat_routes
from backend.managers.factory import create_app_manager
from backend.utils.watcher import start_faq_watcher
from config import FAQ_PATH


app = Flask(__name__, static_folder='frontend', template_folder='frontend')
app.register_blueprint(chat_routes.chat_api, url_prefix='/api')


def bootstrap_manager() -> None:
    """
    Initializes services, managers, and starts background file monitoring.
    """
    # 1. Create managers and attach to the app object
    chat_manager = create_app_manager()
    app.chat_manager = chat_manager
    app.knowledge_manager = chat_manager.knowledge_manager

    # 2. Start FAQ watcher
    start_faq_watcher(FAQ_PATH, chat_manager.knowledge_manager.load_faq_data)

    print("✅ Backend Manager initialized and knowledge base indexed.")


@app.route('/')
def index():
    """Serves the main frontend page."""
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/api/faq/categories')
def get_categories():
    """Returns a list of FAQ categories."""
    return jsonify(app.knowledge_manager.get_categories())


@app.route('/api/faq/questions/<category_id>')
def get_questions(category_id):
    """Returns questions for a specific category."""
    return jsonify(app.knowledge_manager.get_questions_by_category(category_id))


@app.route('/<path:path>')
def static_proxy(path):
    """Proxies static file requests."""
    return send_from_directory(app.static_folder, path)


@app.route('/webhook/telegram', methods=['POST'])
def telegram_webhook():
    chat_manager = app.chat_manager
    data = request.json

    # 1. Handle "Approve" Button Click
    if "callback_query" in data:
        callback = data["callback_query"]
        req_id = callback["data"].replace("approve_", "")

        # Get the original AI suggestion from our state
        request_info = chat_manager.pending_requests.get(req_id)
        if request_info:
            chat_manager.fulfill_request(req_id, request_info["suggestion"])

    # 2. Handle manual Reply
    if "message" in data and "reply_to_message" in data["message"]:
        reply_text = data["message"]["text"]
        original_msg_id = data["message"]["reply_to_message"]["message_id"]
        success = chat_manager.fulfill_by_msg_id(original_msg_id, reply_text)

        if success:
            print(f"✅ Reply mapped to user request via Msg ID {original_msg_id}")
        else:
            print(f"⚠️ Unknown message ID {original_msg_id}")

    return {"status": "ok"}


if __name__ == '__main__':
    bootstrap_manager()
    app.run(debug=False, port=5000)
