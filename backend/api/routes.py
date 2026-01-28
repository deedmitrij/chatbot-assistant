from flask import Blueprint, request, jsonify

chat_api = Blueprint('chat_api', __name__)

# We declare manager as None; main.py will assign the initialized instance to this variable
manager = None


@chat_api.route('/process', methods=['POST'])
def handle_chat():
    global manager
    data = request.json
    user_msg = data.get('message')

    if not user_msg:
        return jsonify({"error": "No message provided"}), 400

    if manager is None:
        return jsonify({"error": "System initializing, please try again in a moment."}), 503

    result = manager.process_message(user_msg)
    return jsonify(result)
