import uuid
from typing import Dict, Any
from backend.managers.knowledge_manager import KnowledgeManager
from backend.services.llm_service import LLMService
from backend.services.vector_db_service import VectorDBService
from backend.services.telegram_service import TelegramService
from config import VECTOR_SIMILARITY_THRESHOLD


class ChatManager:
    """
    Orchestrates the conversation flow, deciding between autonomous AI responses
    and Human-in-the-loop (HITL) Telegram alerts.
    """

    def __init__(self, knowledge_manager: KnowledgeManager):
        self.llm = LLMService()
        self.db = VectorDBService()
        self.tg_service = TelegramService()
        self.knowledge_manager = knowledge_manager
        self.pending_requests = {}  # Stores request details
        self.msg_id_map = {}  # Index for Replies

    def process_message(self, user_query: str) -> Dict[str, Any]:
        """
        Analyzes the user query, searches context, and either returns an answer
        or initiates a human operator request.
        """
        context, distance = self.knowledge_manager.get_relevant_context(user_query)
        ai_answer, is_ai_confident = self.llm.get_answer(user_query, context)

        if distance <= VECTOR_SIMILARITY_THRESHOLD and is_ai_confident:
            return {"status": "direct", "answer": ai_answer}
        else:
            # Create a unique ID for this specific interaction
            req_id = str(uuid.uuid4())

            # Send the alert to Telegram and receive a message ID
            tg_msg_id = self.tg_service.send_alert(
                request_id=req_id,
                user_query=user_query,
                ai_suggestion=ai_answer
            )

            # Store user_query, so we can learn from it later
            self.pending_requests[req_id] = {
                "status": "pending",
                "user_query": user_query,
                "answer": None,
                "suggestion": ai_answer
            }

            # Map the message ID -> UUID connection
            if tg_msg_id:
                self.msg_id_map[tg_msg_id] = req_id

            return {"status": "pending", "request_id": req_id}

    def fulfill_request(self, req_id: str, final_answer: str) -> None:
        """
        Completes a pending request and updates the knowledge base with the verified answer.
        """
        if req_id in self.pending_requests:
            request_data = self.pending_requests[req_id]

            # 1. Save the human-verified answer to the vector database
            self.knowledge_manager.save_operator_answer(
                question=request_data["user_query"],
                answer=final_answer
            )

            # 2. Update status
            request_data["status"] = "completed"
            request_data["answer"] = final_answer
            print(f"✅ Request {req_id} fulfilled and indexed.")

    def fulfill_by_msg_id(self, msg_id: int, final_answer: str) -> bool:
        """
        Matches a Telegram reply to a specific user request using message ID.
        """
        req_id = self.msg_id_map.get(msg_id)
        if req_id:
            self.fulfill_request(req_id, final_answer)
            del self.msg_id_map[msg_id]
            return True
        return False

    def check_status(self, req_id: str):
        """Checks the current status of a pending request for frontend polling."""
        return self.pending_requests.get(req_id, {"status": "not_found"})
