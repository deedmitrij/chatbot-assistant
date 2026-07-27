import json
import requests
from typing import Optional
from config import TG_ADMIN_ID, TG_BOT_TOKEN


class TelegramService:
    """Telegram service for operator interaction."""

    @staticmethod
    def send_alert(request_id: str, user_query: str, ai_suggestion: str) -> Optional[int]:
        """
        Sends an alert to the operator when the AI is uncertain.

        Args:
            request_id (str): Unique request identifier.
            user_query (str): The original text of the user's request.
            ai_suggestion (str): The text of the AI-generated response.

        Returns:
            Optional[int]: ID of the sent message if successful, otherwise None.
        """
        message = (
            f"🚨 **Pending Request**\n\n"
            f"👤 **Guest asked:** {user_query}\n"
            f"🤖 **AI Suggested:** {ai_suggestion}\n"
            "--- \n"
            "👇 **Actions:**\n"
            "1. Click **Approve** to send AI suggestion.\n"
            "2. OR **Reply** to this message to edit."
        )

        # Inline button for instant approval
        keyboard = {
            "inline_keyboard": [[
                {"text": "✅ Approve", "callback_data": f"approve_{request_id}"}
            ]]
        }

        payload = {
            "chat_id": TG_ADMIN_ID,
            "text": message,
            "parse_mode": "Markdown",
            "reply_markup": json.dumps(keyboard)
        }

        try:
            response = requests.post(
                url=f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage",
                json=payload
            )
            resp_data = response.json()
            if resp_data.get("ok"):
                return resp_data["result"]["message_id"]
            else:
                print(f"❌ Telegram Error: {resp_data}")
                return None
        except Exception as e:
            print(f"❌ Telegram Exception: {e}")
            return None

    @staticmethod
    def send_operator_call() -> Optional[int]:
        """
        Sends an alert to the operator when a guest explicitly requests a human.
        """
        message = (
            f"🚨 **Operator Requested**\n\n"
            f"👤 A guest has clicked the 'Contact Operator' button and needs immediate assistance."
        )
        payload = {
            "chat_id": TG_ADMIN_ID,
            "text": message,
            "parse_mode": "Markdown",
        }
        try:
            response = requests.post(
                url=f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage",
                json=payload
            )
            resp_data = response.json()
            if resp_data.get("ok"):
                return resp_data["result"]["message_id"]
            else:
                print(f"❌ Telegram Error sending operator call: {resp_data}")
                return None
        except Exception as e:
            print(f"❌ Telegram Exception in operator call: {e}")
            return None
