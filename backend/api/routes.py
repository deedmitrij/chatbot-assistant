from flask import Blueprint, request, jsonify, current_app


chat_api = Blueprint('chat_api', __name__)


@chat_api.route('/process', methods=['POST'])
def handle_chat():
    chat_manager = getattr(current_app, 'chat_manager', None)
    data = request.json
    user_msg = data.get('message')

    if not user_msg:
        return jsonify({"error": "No message provided"}), 400

    if chat_manager is None:
        return jsonify({"error": "System initializing, please try again in a moment."}), 503

    result = chat_manager.process_message(user_msg)
    return jsonify(result)


@chat_api.route('/check_status/<req_id>', methods=['GET'])
def check_status(req_id):
    status_info = current_app.chat_manager.check_status(req_id)
    return jsonify(status_info)
