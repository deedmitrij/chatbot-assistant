import json

import pytest
from openai import AsyncOpenAI
from ragas.llms import llm_factory

from config import RAG_EVALUATION_DATA_DIR, LLM_BASE_URL, LLM_API_KEY, JUDGE_MODEL
from backend.services.vector_db_service import VectorDBService

# Only the golden files that carry an expected_id/expected_ids (a golden
# document) are usable here — test_stratification.json's HITL/REJECT zone
# cases have no golden document and stay Custom-only.
RETRIEVAL_FILES_WITH_EXPECTED_DOCS = (
    "test_retrieval.json",
    "test_top_k.json",
    "test_metadata.json",
    "test_retrieval_metrics.json",
)


@pytest.fixture(scope="session")
def vector_db_service():
    # Distinct collection name from Custom's "test_collection" so both suites
    # can run in the same pytest session without interfering with each other.
    collection = "ragas_test_collection"
    db = VectorDBService(collection=collection)
    yield db
    db.client.delete_collection(collection)


@pytest.fixture(scope="session")
def ragas_judge_llm():
    """RAGAS evaluator LLM, wired to the same local Ollama Judge model/endpoint
    as the Custom framework's judge (never CHAT_MODEL, never HF)."""
    client = AsyncOpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    return llm_factory(
        model=JUDGE_MODEL,
        provider="openai",
        client=client,
        # Disables Qwen3's "thinking" mode, matching LLMService's judge path.
        extra_body={"reasoning_effort": "none"},
    )


def get_retrieval_ragas_cases():
    """Utility to load RAGAS-eligible retrieval test cases from the shared
    golden data. Same (dataset, case) tuple shape as Custom's own loader."""
    test_cases = []
    for file_name in RETRIEVAL_FILES_WITH_EXPECTED_DOCS:
        file_path = RAG_EVALUATION_DATA_DIR / "retrieval" / file_name
        with open(file_path, "r") as f:
            suites = json.load(f)
        for suite in suites:
            for case in suite["cases"]:
                if "expected_id" in case or "expected_ids" in case:
                    test_cases.append((suite["dataset"], case))
    return test_cases


@pytest.fixture
def test_case(vector_db_service, request):
    """Load the case's dataset into the vector database, then hand back the case."""
    dataset, case = request.param
    existing_ids = vector_db_service.collection.get()["ids"]
    vector_db_service.delete_by_ids(existing_ids)
    vector_db_service.upsert_batch(
        documents=dataset["documents"],
        ids=dataset["ids"],
        metadatas=dataset["metadatas"]
    )
    return case
