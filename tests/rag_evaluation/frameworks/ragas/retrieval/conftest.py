import json
import pytest
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from config import PROJECT_ROOT, RAG_EVALUATION_DATA_DIR, LLM_BASE_URL, LLM_API_KEY, JUDGE_MODEL
from backend.services.vector_db_service import VectorDBService
from tests.rag_evaluation.reporting.recorder import Recorder
from tests.rag_evaluation.reporting.aggregate import write_latest
from tests.rag_evaluation.reporting.dimensions import get_quality_dimension
from tests.rag_evaluation.reporting.schema import Status

EVENTS_PATH = PROJECT_ROOT / "reports" / "results" / "events.jsonl"
LATEST_PATH = PROJECT_ROOT / "reports" / "results" / "latest.json"


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
                    case["case_id"] = f"{file_name}::{case['name']}"
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


@pytest.fixture(scope="session")
def recorder():
    """One Recorder per pytest session, so every metric module invoked in
    the same run shares one run_id. Persists to the same events.jsonl the
    reporting subsystem's aggregator reads from — same pattern as
    Generation's conftest.py (see ../generation/conftest.py)."""
    return Recorder(EVENTS_PATH)


def search_retrieval(vector_db_service, test_case):
    """One ChromaDB search per case, same query/top_k/filter as before.
    Returns (retrieved_contexts, retrieved_ids) from that single search
    result, so the RAGAS metric input and the reporting fields can never
    diverge by coming from two separate searches."""
    n_results = test_case.get("n_results", 3)
    search_result = vector_db_service.search(
        query_text=test_case["query"],
        n_results=n_results,
        where_filter=test_case.get("filter"),
    )
    return search_result["documents"][0], search_result["ids"][0]


def _expected_ids(test_case):
    """Normalizes the golden data's expected_id (singular) / expected_ids
    (list) shapes into one list, for the reporting-only expected_ids field."""
    if "expected_ids" in test_case:
        return test_case["expected_ids"]
    if "expected_id" in test_case:
        return [test_case["expected_id"]]
    return None


def record_retrieval_result(*, recorder, test_case, metric_name, result, retrieved_contexts, retrieved_ids):
    """Reporting-only: persists one EvaluationResult for an already-computed
    RAGAS MetricResult, then re-materializes latest.json. Mirrors
    Generation's record_generation_result (see ../generation/conftest.py)
    exactly, adapted for Retrieval's fields — never constructs a metric,
    never calls .score(...), and never alters the result.

    Status: NaN -> N/A, score=None, threshold=None; finite value with
    min_score configured -> PASS/FAIL; finite value with no min_score yet
    -> BASELINE (not currently exercised — every collected Retrieval case
    already has an approved min_score, but this keeps the same generic
    contract as Generation's reporting helper).

    No Assistant model is involved in Retrieval at all (no LLM generation
    step here, only vector search + Judge scoring), so assistant_model and
    embedding_model are always None — recording CHAT_MODEL here would be
    an incorrect attribution.
    """
    is_nan = result.value != result.value
    min_score = test_case["evaluation"]["ragas"][metric_name].get("min_score")

    if is_nan:
        status = Status.NOT_APPLICABLE
        score = None
        threshold = None
    else:
        score = float(result.value)
        if min_score is None:
            status = Status.BASELINE
            threshold = None
        else:
            status = Status.PASS if score >= min_score else Status.FAIL
            threshold = min_score

    recorder.record(
        framework="ragas",
        phase="retrieval",
        suite=test_case["case_id"].split("::", 1)[0],
        case_id=test_case["case_id"],
        case_name=test_case["name"],
        metric=metric_name,
        quality_dimension=get_quality_dimension("ragas", metric_name),
        query=test_case["query"],
        threshold=threshold,
        status=status,
        score=score,
        retrieved_contexts=retrieved_contexts,
        expected_ids=_expected_ids(test_case),
        retrieved_ids=retrieved_ids,
        filter=test_case.get("filter"),
        reference_answer=test_case.get("reference_answer"),
        judge_reason=getattr(result, "reason", None),
        assistant_model=None,
        judge_model=JUDGE_MODEL,
        embedding_model=None,
    )
    write_latest(EVENTS_PATH, LATEST_PATH)
