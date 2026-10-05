import pytest
from config import JUDGE_MODEL
from backend.services.vector_db_service import VectorDBService
from reporting.paths import EVENTS_PATH, LATEST_PATH
from reporting.aggregate import write_latest
from reporting.dimensions import get_quality_dimension
from reporting.schema import Status
from tests.rag_evaluation.frameworks.deepeval.conftest import resolve_min_score


@pytest.fixture(scope="session")
def vector_db_service():
    # Distinct collection name from Custom's "test_collection" and RAGAS's
    # "ragas_test_collection" so all three suites can run in the same pytest
    # session without interfering with each other.
    collection = "deepeval_test_collection"
    db = VectorDBService(collection=collection)
    yield db
    db.client.delete_collection(collection)


@pytest.fixture
def test_case(vector_db_service, request):
    """Load the case's dataset into the vector database, then hand back the case."""
    case = request.param
    dataset = case["dataset"]
    existing_ids = vector_db_service.collection.get()["ids"]
    vector_db_service.delete_by_ids(existing_ids)
    vector_db_service.upsert_batch(
        documents=dataset["documents"],
        ids=dataset["ids"],
        metadatas=dataset["metadatas"]
    )
    return case


def search_retrieval(vector_db_service, test_case):
    """One ChromaDB search per case. Returns (retrieved_contexts,
    retrieved_ids) from a single search result."""
    n_results = test_case.get("n_results", 3)
    search_result = vector_db_service.search(
        query_text=test_case["query"],
        n_results=n_results,
        where_filter=test_case.get("filter"),
    )
    return search_result["documents"][0], search_result["ids"][0]


def _expected_ids(test_case):
    """Normalizes expected_id (singular) / expected_ids (list) into one list."""
    if "expected_ids" in test_case:
        return test_case["expected_ids"]
    if "expected_id" in test_case:
        return [test_case["expected_id"]]
    return None


def record_retrieval_result(*, recorder, test_case, metric_name, score, reason, retrieved_contexts, retrieved_ids):
    """Persists one EvaluationResult for an already-measured DeepEval
    metric's score/reason. Never constructs or measures the metric itself.

    No Assistant model is involved in Retrieval, so assistant_model and
    embedding_model are always None.
    """
    is_nan = score != score
    min_score = resolve_min_score(test_case["evaluation"]["deepeval"], metric_name)

    if is_nan:
        status = Status.NOT_APPLICABLE
        recorded_score = None
        threshold = None
    else:
        recorded_score = float(score)
        if min_score is None:
            status = Status.BASELINE
            threshold = None
        else:
            status = Status.PASS if recorded_score >= min_score else Status.FAIL
            threshold = min_score

    recorder.record(
        framework="deepeval",
        phase="retrieval",
        suite=test_case["case_id"].split("::", 1)[0],
        case_id=test_case["case_id"],
        case_name=test_case["name"],
        metric=metric_name,
        quality_dimension=get_quality_dimension("deepeval", metric_name),
        query=test_case["query"],
        threshold=threshold,
        status=status,
        score=recorded_score,
        retrieved_contexts=retrieved_contexts,
        expected_ids=_expected_ids(test_case),
        retrieved_ids=retrieved_ids,
        filter=test_case.get("filter"),
        reference_answer=test_case.get("reference_answer"),
        judge_reason=reason,
        assistant_model=None,
        judge_model=JUDGE_MODEL,
        embedding_model=None,
    )
    write_latest(EVENTS_PATH, LATEST_PATH)
