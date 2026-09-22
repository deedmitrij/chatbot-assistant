import json
import pytest
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from config import RAG_EVALUATION_DATA_DIR, LLM_BASE_URL, LLM_API_KEY, JUDGE_MODEL
from backend.services.vector_db_service import VectorDBService


# Only the golden files that carry an expected_id/expected_ids (a golden
# document) are usable here — distance_stratification.json's HITL/REJECT
# zone cases have no golden document and stay Custom-only.
RETRIEVAL_FILES_WITH_EXPECTED_DOCS = (
    "query_robustness.json",
    "top_k_retrieval.json",
    "metadata_filtering.json",
    "ranking_metrics.json",
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
    """RAGAS evaluator LLM, wired to the same local Ollama Judge model/endpoint as the Custom framework's judge."""
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
    golden data. Same (dataset, case) tuple shape as Custom's own loader.
    Used by ContextRelevance, which needs no reference answer."""
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


def get_single_reference_ragas_cases():
    """Subset of get_retrieval_ragas_cases() whose reference_answer is a
    single, naturally supported factual target — used by
    ContextPrecisionWithReference and ContextRecall, which judge each
    retrieved chunk's support for the *whole* reference answer. Cases marked
    reference_answer_is_composite (a reference synthesizing facts from more
    than one expected document) are excluded: no single chunk can support a
    multi-fact reference in full, so these metrics score them near-zero
    regardless of retrieval quality. Those cases stay fully covered by
    Custom's ranking metrics and by ContextRelevance."""
    return [
        (dataset, case)
        for dataset, case in get_retrieval_ragas_cases()
        if not case.get("evaluation", {}).get("reference_answer_is_composite", False)
    ]


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


def _retrieved_contexts(vector_db_service, test_case):
    n_results = test_case.get("n_results", 3)
    search_result = vector_db_service.search(
        query_text=test_case["query"],
        n_results=n_results,
        where_filter=test_case.get("filter"),
    )
    return search_result["documents"][0]
