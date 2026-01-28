import json
from flask import Flask, send_from_directory, jsonify, request
from backend.manager import ChatManager
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

        # 1. Create the manager instance with the token
        new_manager = ChatManager()

        # 2. Build the FAISS index
        new_manager.init_knowledge(texts)

        # 3. Inject it into the routes module so the API can use it
        chat_routes.manager = new_manager
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


if __name__ == '__main__':
    bootstrap_manager()
    app.run(debug=False, port=5000)
