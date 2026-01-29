from flask import Blueprint, request, jsonify, current_app


chat_api = Blueprint('chat_api', __name__)


@chat_api.route('/process', methods=['POST'])
def handle_chat():
    manager = getattr(current_app, 'manager', None)
    data = request.json
    user_msg = data.get('message')

    if not user_msg:
        return jsonify({"error": "No message provided"}), 400

    if manager is None:
        return jsonify({"error": "System initializing, please try again in a moment."}), 503

    result = manager.process_message(user_msg)
    return jsonify(result)


@chat_api.route('/check_status/<req_id>', methods=['GET'])
def check_status(req_id):
    manager = current_app.manager
    status_info = manager.check_status(req_id)
    return jsonify(status_info)
