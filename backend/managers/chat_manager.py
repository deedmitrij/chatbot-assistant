import uuid
from typing import Dict, Any, Optional
from backend.managers.knowledge_manager import KnowledgeManager
from backend.services.llm.llm_service import LLMService
from backend.services.vector_db_service import VectorDBService
from backend.services.telegram_service import TelegramService
from backend.services.database_service import DatabaseService
from config import VECTOR_SIMILARITY_THRESHOLD


class ChatManager:
    """
    Orchestrates the conversation flow, deciding between autonomous AI responses
    and Human-in-the-loop (HITL) Telegram alerts.
    """

    def __init__(
        self,
        knowledge_manager: KnowledgeManager,
        db_service: Optional[DatabaseService] = None,
        llm_service: Optional[LLMService] = None,
        vector_db_service: Optional[VectorDBService] = None,
        telegram_service: Optional[TelegramService] = None,
    ):
        from config import DB_PATH
        self.llm = llm_service or LLMService()
        self.db = vector_db_service or VectorDBService()
        self.tg_service = telegram_service or TelegramService()
        self.knowledge_manager = knowledge_manager
        self.db_service = db_service or DatabaseService(DB_PATH)

    def process_message(self, user_query: str) -> Dict[str, Any]:
        """
        Analyzes the user query, searches context, and either returns an answer
        or initiates a human operator request.
        """
        vector_search_result = self.knowledge_manager.get_relevant_context(user_query)

        context = vector_search_result['documents'][0]
        nearest_distance = vector_search_result['distances'][0][0]

        ai_response = self.llm.get_answer(user_query, context)
        is_ai_confident = ai_response['confidence']
        ai_answer = ai_response['answer']

        if nearest_distance <= VECTOR_SIMILARITY_THRESHOLD and is_ai_confident:
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

            # Store user_query and details in database
            self.db_service.create_request(
                req_id=req_id,
                user_query=user_query,
                suggestion=ai_answer,
                tg_msg_id=tg_msg_id
            )

            return {"status": "pending", "request_id": req_id}

    def process_message_for_eval(self, user_query: str) -> Dict[str, Any]:
        """
        Evaluation-only contract: returns the exact context used and the answer — just data for RAGAS.
        """
        context, distance = self.knowledge_manager.get_relevant_context(user_query)
        ai_answer, is_ai_confident = self.llm.get_answer(user_query, context)

        return {"answer": ai_answer, "context": context}

    def fulfill_request(self, req_id: str, final_answer: str) -> None:
        """
        Completes a pending request and updates the knowledge base with the verified answer.
        """
        request_data = self.db_service.get_request(req_id)
        if request_data:
            # 1. Save the human-verified answer to the vector database
            self.knowledge_manager.save_operator_answer(
                question=request_data["user_query"],
                answer=final_answer
            )

            # 2. Update status and answer in the database
            self.db_service.update_request_fulfillment(req_id, final_answer)
            print(f"✅ Request {req_id} fulfilled and indexed.")

    def fulfill_by_msg_id(self, msg_id: int, final_answer: str) -> bool:
        """
        Matches a Telegram reply to a specific user request using message ID.
        """
        request_data = self.db_service.get_request_by_tg_msg_id(msg_id)
        if request_data:
            self.fulfill_request(request_data["id"], final_answer)
            return True
        return False

    def check_status(self, req_id: str):
        """Checks the current status of a pending request for frontend polling."""
        request_data = self.db_service.get_request(req_id)
        if request_data:
            return request_data
        return {"status": "not_found"}
