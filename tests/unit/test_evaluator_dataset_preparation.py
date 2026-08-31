from backend.evaluation.evaluator import RAGEvaluator


class FakeChatManagerForEval:
    """
    Duck-typed stand-in for ChatManager. Only implements process_message_for_eval,
    keyed by question, so no real LLM/vector DB/Telegram/SQLite is ever touched.
    """

    def __init__(self, results_by_question):
        self._results_by_question = results_by_question
        self.calls = []

    def process_message_for_eval(self, query):
        self.calls.append(query)
        return self._results_by_question[query]


def _make_evaluator(chat_manager):
    """
    Build a RAGEvaluator without running RAGEvaluator.__init__, which would
    otherwise construct real AsyncOpenAI/HuggingFaceEmbeddings clients (network
    calls / model downloads). prepare_ragas_dataset only needs self.chat_manager.
    """
    evaluator = RAGEvaluator.__new__(RAGEvaluator)
    evaluator.chat_manager = chat_manager
    return evaluator


def test_prepare_ragas_dataset_keeps_multi_document_context_flat():
    """
    Regression test: process_message_for_eval now returns
    {"answer": ..., "context": ["doc1", "doc2", "doc3"]}.
    RAGAS expects dataset["contexts"][i] to be that same flat list of chunk
    strings for question i, not a list wrapping that list again.
    """
    test_data = [
        {"question": "When can I check in?", "ground_truth": "Check-in is at 3:00 PM."},
        {"question": "Is the pool heated?", "ground_truth": "Yes, heated to 28C."},
    ]
    fake_chat_manager = FakeChatManagerForEval({
        "When can I check in?": {
            "answer": "Check-in is at 3:00 PM.",
            "context": ["doc1", "doc2", "doc3"],
        },
        "Is the pool heated?": {
            "answer": "Yes, heated to 28C.",
            "context": ["doc4"],
        },
    })
    evaluator = _make_evaluator(fake_chat_manager)

    dataset = evaluator.prepare_ragas_dataset(test_data)

    contexts = dataset["contexts"]
    assert contexts == [["doc1", "doc2", "doc3"], ["doc4"]], (
        "RAGAS expects contexts[i] to be the flat list of retrieved chunk "
        "strings for question i, not a list wrapping that list again."
    )
    assert dataset["answer"] == ["Check-in is at 3:00 PM.", "Yes, heated to 28C."]
    assert dataset["question"] == ["When can I check in?", "Is the pool heated?"]
    assert dataset["ground_truth"] == ["Check-in is at 3:00 PM.", "Yes, heated to 28C."]
