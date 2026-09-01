import uuid

import backend.managers.chat_manager as chat_manager_module
from backend.managers.chat_manager import ChatManager

# Read the threshold actually bound into ChatManager's module namespace (it's
# imported by value: `from config import VECTOR_SIMILARITY_THRESHOLD`), rather
# than hardcoding a number, so these tests stay correct if .env ever changes it.
THRESHOLD = chat_manager_module.VECTOR_SIMILARITY_THRESHOLD
NEAR_DISTANCE = max(THRESHOLD - 0.1, 0.0)
FAR_DISTANCE = THRESHOLD + 0.1


class FakeKnowledgeManager:
    """Duck-typed stand-in returning a realistic Chroma-shaped retrieval result."""

    def __init__(self, retrieval_result):
        self._retrieval_result = retrieval_result
        self.calls = []

    def get_relevant_context(self, query):
        self.calls.append(query)
        return self._retrieval_result


class FakeLLMService:
    """Duck-typed stand-in returning the current get_answer {confidence, answer} dict contract."""

    def __init__(self, response):
        self._response = response
        self.calls = []

    def get_answer(self, query, context):
        self.calls.append((query, context))
        return self._response


class FakeTelegramService:
    """Duck-typed stand-in for TelegramService.send_alert. No real Telegram/network call."""

    def __init__(self, tg_msg_id=555):
        self._tg_msg_id = tg_msg_id
        self.alerts = []

    def send_alert(self, request_id, user_query, ai_suggestion):
        self.alerts.append({
            "request_id": request_id,
            "user_query": user_query,
            "ai_suggestion": ai_suggestion,
        })
        return self._tg_msg_id


class FakeDatabaseService:
    """Duck-typed stand-in for DatabaseService.create_request. No real SQLite involved."""

    def __init__(self):
        self.created_requests = []

    def create_request(self, req_id, user_query, suggestion, tg_msg_id):
        self.created_requests.append({
            "req_id": req_id,
            "user_query": user_query,
            "suggestion": suggestion,
            "tg_msg_id": tg_msg_id,
        })


class Unused:
    """
    Sentinel for collaborators a given path must never touch. Any attribute
    access fails the test loudly instead of silently constructing/using a
    real service.
    """

    def __getattr__(self, name):
        raise AssertionError(f"this path must not access '{name}'")


def _retrieval_result(distance, document="The pool is open from 8 AM to 10 PM."):
    """Realistic single-match Chroma-shaped query() result."""
    return {
        "ids": [["doc_pool"]],
        "documents": [[document]],
        "distances": [[distance]],
        "metadatas": [[{"source": "faq"}]],
    }


def _make_chat_manager(knowledge_manager, llm_service, db_service=None, telegram_service=None):
    """
    ChatManager wired only to fakes/sentinels. vector_db_service is passed an
    Unused() sentinel because process_message never touches self.db directly
    (it goes through knowledge_manager.get_relevant_context instead), and
    ChatManager.__init__ would otherwise construct a real VectorDBService.
    """
    return ChatManager(
        knowledge_manager=knowledge_manager,
        llm_service=llm_service,
        db_service=db_service if db_service is not None else Unused(),
        vector_db_service=Unused(),
        telegram_service=telegram_service if telegram_service is not None else Unused(),
    )


# --- A. near retrieval + confidence=True -> direct answer -----------------

def test_near_distance_and_confident_returns_direct_answer():
    knowledge_manager = FakeKnowledgeManager(_retrieval_result(NEAR_DISTANCE))
    llm_service = FakeLLMService({"confidence": True, "answer": "Yes, we have a pool."})
    chat_manager = _make_chat_manager(knowledge_manager, llm_service)

    result = chat_manager.process_message("Do you have a pool?")

    assert result == {"status": "direct", "answer": "Yes, we have a pool."}
    assert knowledge_manager.calls == ["Do you have a pool?"]
    assert llm_service.calls == [("Do you have a pool?", ["The pool is open from 8 AM to 10 PM."])]


def test_near_distance_and_confident_does_not_touch_db_or_telegram():
    """
    The direct-answer path must not create a pending DB request or send a
    Telegram HITL notification. db_service/telegram_service are Unused()
    sentinels here, so any access to create_request/send_alert fails loudly.
    """
    knowledge_manager = FakeKnowledgeManager(_retrieval_result(NEAR_DISTANCE))
    llm_service = FakeLLMService({"confidence": True, "answer": "Yes, we have a pool."})
    chat_manager = _make_chat_manager(knowledge_manager, llm_service)

    chat_manager.process_message("Do you have a pool?")  # would raise via Unused if touched


# --- B. near retrieval + confidence=False -> HITL/pending -----------------

def test_near_distance_and_not_confident_returns_pending():
    knowledge_manager = FakeKnowledgeManager(_retrieval_result(NEAR_DISTANCE))
    llm_service = FakeLLMService({"confidence": False, "answer": "I'm not fully sure about that."})
    fake_tg = FakeTelegramService(tg_msg_id=111)
    fake_db = FakeDatabaseService()
    chat_manager = _make_chat_manager(knowledge_manager, llm_service, fake_db, fake_tg)

    result = chat_manager.process_message("Do you have a pool?")

    assert result["status"] == "pending"
    request_id = result["request_id"]
    uuid.UUID(request_id)  # must be a valid UUID string

    assert fake_tg.alerts == [{
        "request_id": request_id,
        "user_query": "Do you have a pool?",
        "ai_suggestion": "I'm not fully sure about that.",
    }]
    assert fake_db.created_requests == [{
        "req_id": request_id,
        "user_query": "Do you have a pool?",
        "suggestion": "I'm not fully sure about that.",
        "tg_msg_id": 111,
    }]


# --- C. far retrieval + confidence=True -> HITL/pending -------------------

def test_far_distance_and_confident_returns_pending():
    knowledge_manager = FakeKnowledgeManager(_retrieval_result(FAR_DISTANCE))
    llm_service = FakeLLMService({"confidence": True, "answer": "I think we allow pets."})
    fake_tg = FakeTelegramService(tg_msg_id=222)
    fake_db = FakeDatabaseService()
    chat_manager = _make_chat_manager(knowledge_manager, llm_service, fake_db, fake_tg)

    result = chat_manager.process_message("Can I bring my pet elephant?")

    assert result["status"] == "pending"
    request_id = result["request_id"]
    uuid.UUID(request_id)

    assert fake_tg.alerts == [{
        "request_id": request_id,
        "user_query": "Can I bring my pet elephant?",
        "ai_suggestion": "I think we allow pets.",
    }]
    assert fake_db.created_requests == [{
        "req_id": request_id,
        "user_query": "Can I bring my pet elephant?",
        "suggestion": "I think we allow pets.",
        "tg_msg_id": 222,
    }]


# --- D. far retrieval + confidence=False -> HITL/pending ------------------

def test_far_distance_and_not_confident_returns_pending():
    knowledge_manager = FakeKnowledgeManager(_retrieval_result(FAR_DISTANCE))
    llm_service = FakeLLMService({"confidence": False, "answer": "I don't have information about that."})
    fake_tg = FakeTelegramService(tg_msg_id=333)
    fake_db = FakeDatabaseService()
    chat_manager = _make_chat_manager(knowledge_manager, llm_service, fake_db, fake_tg)

    result = chat_manager.process_message("Do you have a helipad?")

    assert result["status"] == "pending"
    request_id = result["request_id"]
    uuid.UUID(request_id)

    assert fake_tg.alerts == [{
        "request_id": request_id,
        "user_query": "Do you have a helipad?",
        "ai_suggestion": "I don't have information about that.",
    }]
    assert fake_db.created_requests == [{
        "req_id": request_id,
        "user_query": "Do you have a helipad?",
        "suggestion": "I don't have information about that.",
        "tg_msg_id": 333,
    }]


# --- Boundary: distance exactly at the threshold ---------------------------

def test_distance_exactly_at_threshold_and_confident_returns_direct_answer():
    """
    process_message uses `nearest_distance <= VECTOR_SIMILARITY_THRESHOLD`,
    so a distance exactly equal to the threshold is inclusive (direct-answer
    eligible), not routed to HITL.
    """
    knowledge_manager = FakeKnowledgeManager(_retrieval_result(THRESHOLD))
    llm_service = FakeLLMService({"confidence": True, "answer": "Yes, we have a pool."})
    chat_manager = _make_chat_manager(knowledge_manager, llm_service)

    result = chat_manager.process_message("Do you have a pool?")

    assert result == {"status": "direct", "answer": "Yes, we have a pool."}
