import json
import pytest
from backend.managers import knowledge_manager as km_module
from backend.managers.knowledge_manager import KnowledgeManager


class FakeVectorDBService:
    """Stand-in for VectorDBService so this unit test never touches a real ChromaDB."""

    def __init__(self, *args, **kwargs):
        self.upserted = []

    def upsert_batch(self, documents, ids, metadatas):
        self.upserted.append((documents, ids, metadatas))

    def get_ids_by_metadata(self, filter_dict):
        return []

    def delete_by_ids(self, ids):
        pass


def _write_operator_knowledge(path, entries):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=4, ensure_ascii=False)


def _read_operator_knowledge(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def knowledge_manager(monkeypatch, tmp_path):
    """
    KnowledgeManager wired to a temp operator_knowledge.json and a fake vector DB.
    No real ChromaDB or other external service is touched, and the real
    operator_knowledge.json in the repo root is never opened.
    """
    operator_path = tmp_path / "operator_knowledge.json"
    monkeypatch.setattr(km_module, "VectorDBService", FakeVectorDBService)
    monkeypatch.setattr(km_module, "OPERATOR_KNOWLEDGE_PATH", operator_path)

    manager = KnowledgeManager()
    return manager, operator_path


def test_save_operator_answer_updates_existing_entry_at_index_zero(knowledge_manager):
    manager, operator_path = knowledge_manager
    _write_operator_knowledge(operator_path, [
        {"q": "Are you pet friendly?", "a": "old answer", "created_at": "2026-01-01 00:00:00"},
        {"q": "Is parking free?", "a": "Yes", "created_at": "2026-01-01 00:00:00"},
    ])

    manager.save_operator_answer(question="Are you pet friendly?", answer="new answer")

    data = _read_operator_knowledge(operator_path)
    assert len(data) == 2, "Existing index-0 entry must be replaced, not duplicated"
    assert data[0]["q"] == "Are you pet friendly?"
    assert data[0]["a"] == "new answer"


def test_save_operator_answer_updates_existing_entry_at_nonzero_index(knowledge_manager):
    manager, operator_path = knowledge_manager
    _write_operator_knowledge(operator_path, [
        {"q": "Are you pet friendly?", "a": "Yes", "created_at": "2026-01-01 00:00:00"},
        {"q": "Is parking free?", "a": "old answer", "created_at": "2026-01-01 00:00:00"},
    ])

    manager.save_operator_answer(question="Is parking free?", answer="new answer")

    data = _read_operator_knowledge(operator_path)
    assert len(data) == 2, "Existing non-zero-index entry must be replaced, not duplicated"
    assert data[1]["q"] == "Is parking free?"
    assert data[1]["a"] == "new answer"


def test_save_operator_answer_appends_when_question_not_found(knowledge_manager):
    manager, operator_path = knowledge_manager
    _write_operator_knowledge(operator_path, [
        {"q": "Are you pet friendly?", "a": "Yes", "created_at": "2026-01-01 00:00:00"},
    ])

    manager.save_operator_answer(question="Do you have a spa?", answer="Yes, on the 2nd floor")

    data = _read_operator_knowledge(operator_path)
    assert len(data) == 2
    assert data[-1]["q"] == "Do you have a spa?"
    assert data[-1]["a"] == "Yes, on the 2nd floor"
