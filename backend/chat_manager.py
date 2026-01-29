import uuid
from backend.services.llm_service import LLMService
from backend.services.vector_db_service import VectorDBService
from backend.services.telegram_service import TelegramService
from config import VECTOR_SIMILARITY_THRESHOLD


class ChatManager:
    def __init__(self):
        self.llm = LLMService()
        self.db = VectorDBService()
        self.tg_service = TelegramService()
        self.pending_requests = {}  # Stores request details
        self.msg_id_map = {}  # Index for Replies

    def init_knowledge(self, raw_texts: list):
        print("Indexing knowledge...")
        vectors = self.llm.embed_content(content=raw_texts)
        self.db.store(raw_texts, vectors)
        print("✅ Knowledge Base is ready.")

    def process_message(self, user_query: str):
        query_vector = self.llm.embed_content(user_query, is_query=True)
        context, confidence = self.db.search(query_vector)
        ai_answer = self.llm.get_answer(user_query, context)

        if confidence >= VECTOR_SIMILARITY_THRESHOLD:
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

            # Save the request
            self.pending_requests[req_id] = {
                "status": "pending",
                "answer": None,
                "suggestion": ai_answer
            }

            # Map the message ID -> UUID connection
            if tg_msg_id:
                self.msg_id_map[tg_msg_id] = req_id

            return {"status": "pending", "request_id": req_id}

    def fulfill_request(self, req_id: str, final_answer: str):
        """Closes a request by request ID (for the 'Approve')"""
        if req_id in self.pending_requests:
            self.pending_requests[req_id]["status"] = "completed"
            self.pending_requests[req_id]["answer"] = final_answer
            print(f"✅ Request {req_id} fulfilled.")

    def fulfill_by_msg_id(self, msg_id: int, final_answer: str):
        """Closes a request by request ID (for the 'Reply')"""
        req_id = self.msg_id_map.get(msg_id)
        if req_id:
            self.fulfill_request(req_id, final_answer)
            del self.msg_id_map[msg_id]
            return True
        return False

    def check_status(self, req_id: str):
        """Called by the Frontend (Polling)."""
        return self.pending_requests.get(req_id, {"status": "not_found"})
