from backend.services.llm_service import LLMService
from backend.services.vector_db_service import VectorDBService
from config import VECTOR_SIMILARITY_THRESHOLD

#
# class ChatManager:
#     def __init__(self, hf_token, threshold=0.65):
#         self.vector_db = VectorDBService()
#         self.llm_svc = LLMService()
#         self.threshold = threshold
#
#     def init_knowledge(self, texts):
#         self.vector_db.build_index(texts)
#
#     def process_message(self, message):
#         context, confidence = self.vector_db.query(message)
#         suggested_answer = self.llm_svc.get_answer(message, context)
#
#         if confidence >= self.threshold:
#             return {"status": "direct", "answer": suggested_answer}
#         else:
#             return {
#                 "status": "pending_approval",
#                 "answer": "Let me double check that with our front desk staff...",
#                 "suggested": suggested_answer
#             }


class ChatManager:
    def __init__(self):
        self.llm = LLMService()
        self.db = VectorDBService()
        self.threshold = VECTOR_SIMILARITY_THRESHOLD

    def init_knowledge(self, raw_texts: list):
        print("Indexing knowledge...")
        vectors = self.llm.embed_content(content=raw_texts)
        self.db.store(raw_texts, vectors)
        print("✅ Knowledge Base is ready.")

    def process_message(self, user_query: str):
        # 1. Embed user request
        query_vector = self.llm.embed_content(user_query, is_query=True)

        # 2. Looking for relevant context
        context, confidence = self.db.search(query_vector)

        # 3. Generating a response by LLM
        ai_answer = self.llm.get_answer(user_query, context)

        # 4. Confidence Logic
        if confidence >= VECTOR_SIMILARITY_THRESHOLD:
            return {"status": "direct", "answer": ai_answer}
        else:
            return {
                "status": "pending_approval",
                "answer": "Let me double check that with our front desk staff...",
                "suggested": ai_answer,
                "confidence": float(confidence)
            }
