from backend.managers.chat_manager import ChatManager


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


class Unused:
    """
    Sentinel for collaborators process_message_for_eval must never touch.
    Any attribute access fails the test loudly instead of silently
    constructing/using a real service.
    """

    def __getattr__(self, name):
        raise AssertionError(f"process_message_for_eval must not access '{name}'")


def _make_chat_manager(knowledge_manager, llm_service):
    """
    ChatManager wired only to fakes. db_service/vector_db_service/telegram_service
    are never used by process_message_for_eval, but ChatManager.__init__ builds
    real instances by default when a collaborator isn't passed, so explicit
    truthy sentinels are passed here to guarantee no real SQLite/ChromaDB/
    Telegram service is ever constructed for this test.
    """
    return ChatManager(
        knowledge_manager=knowledge_manager,
        llm_service=llm_service,
        db_service=Unused(),
        vector_db_service=Unused(),
        telegram_service=Unused(),
    )


def test_process_message_for_eval_with_multiple_context_documents():
    retrieval_result = {
        "ids": [["faq_1", "faq_2", "operator_3"]],
        "documents": [[
            "Check-in is at 3:00 PM and check-out is at 11:00 AM.",
            "Our infinity pool is heated to a comfortable 28°C year-round.",
            "Yes, we allow common pets like dogs and cats.",
        ]],
        "distances": [[0.12, 0.34, 0.56]],
        "metadatas": [[{"source": "faq"}, {"source": "faq"}, {"source": "operator"}]],
    }
    llm_response = {"confidence": True, "answer": "Check-in is at 3:00 PM."}

    knowledge_manager = FakeKnowledgeManager(retrieval_result)
    llm_service = FakeLLMService(llm_response)
    chat_manager = _make_chat_manager(knowledge_manager, llm_service)

    result = chat_manager.process_message_for_eval("When can I check in?")

    assert result == {
        "answer": "Check-in is at 3:00 PM.",
        "context": retrieval_result["documents"][0],
    }
    # Confirm the LLM received the actual retrieved document texts, not raw dict keys/values.
    assert llm_service.calls == [("When can I check in?", retrieval_result["documents"][0])]
    assert knowledge_manager.calls == ["When can I check in?"]


def test_process_message_for_eval_with_empty_context():
    retrieval_result = {
        "ids": [[]],
        "documents": [[]],
        "distances": [[]],
        "metadatas": [[]],
    }
    llm_response = {"confidence": False, "answer": "I don't have information about that."}

    knowledge_manager = FakeKnowledgeManager(retrieval_result)
    llm_service = FakeLLMService(llm_response)
    chat_manager = _make_chat_manager(knowledge_manager, llm_service)

    result = chat_manager.process_message_for_eval("Do you have a helipad?")

    assert result == {
        "answer": "I don't have information about that.",
        "context": [],
    }
    assert llm_service.calls == [("Do you have a helipad?", [])]
