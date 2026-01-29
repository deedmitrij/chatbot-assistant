import json
from flask import Flask, send_from_directory, jsonify, request, current_app
from backend.chat_manager import ChatManager
from backend.api import routes as chat_routes


app = Flask(__name__, static_folder='frontend', template_folder='frontend')
app.register_blueprint(chat_routes.chat_api, url_prefix='/api')


def bootstrap_manager():
    with open('knowledge_base.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        texts = []
        for cat_id, questions in data['faq'].items():
            for item in questions:
                texts.append(f"Question: {item['q']} Answer: {item['a']}")

        # 1. Create the manager instance
        manager = ChatManager()

        # 2. Build the FAISS index
        manager.init_knowledge(texts)

        # 3. Attach the manager to the app object
        app.manager = manager
        print("✅ Backend Manager initialized and knowledge base indexed.")


@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/api/faq/categories')
def get_categories():
    # ADD encoding='utf-8' HERE
    with open('knowledge_base.json', 'r', encoding='utf-8') as f:
        return jsonify(json.load(f)['categories'])


@app.route('/api/faq/questions/<category_id>')
def get_questions(category_id):
    # ADD encoding='utf-8' HERE
    with open('knowledge_base.json', 'r', encoding='utf-8') as f:
        return jsonify(json.load(f)['faq'].get(category_id, []))


@app.route('/<path:path>')
def static_proxy(path):
    return send_from_directory(app.static_folder, path)


@app.route('/webhook/telegram', methods=['POST'])
def telegram_webhook():
    manager = current_app.manager
    data = request.json

    # 1. Handle "Approve" Button Click
    if "callback_query" in data:
        callback = data["callback_query"]
        req_id = callback["data"].replace("approve_", "")

        # Get the original AI suggestion from our state
        request_info = manager.pending_requests.get(req_id)
        if request_info:
            manager.fulfill_request(req_id, request_info["suggestion"])

    # 2. Handle manual Reply
    if "message" in data and "reply_to_message" in data["message"]:
        reply_text = data["message"]["text"]
        original_msg_id = data["message"]["reply_to_message"]["message_id"]
        success = manager.fulfill_by_msg_id(original_msg_id, reply_text)

        if success:
            print(f"✅ Reply mapped to user request via Msg ID {original_msg_id}")
        else:
            print(f"⚠️ Unknown message ID {original_msg_id}")

    return {"status": "ok"}


if __name__ == '__main__':
    bootstrap_manager()
    app.run(debug=False, port=5000)
