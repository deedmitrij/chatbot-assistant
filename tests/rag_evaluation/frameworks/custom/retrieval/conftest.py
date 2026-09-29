import pytest
from backend.services.vector_db_service import VectorDBService
from tests.rag_evaluation.reporting.paths import EVENTS_PATH, LATEST_PATH
from tests.rag_evaluation.reporting.aggregate import write_latest
from tests.rag_evaluation.reporting.dimensions import get_quality_dimension
from tests.rag_evaluation.reporting.schema import Status


@pytest.fixture(scope="session")
def vector_db_service():
    collection = "test_collection"
    db = VectorDBService(collection=collection)
    yield db
    db.client.delete_collection(collection)

@pytest.fixture
def test_case(vector_db_service, request):
    """
    Load the dataset to the vector database.
    Pass the test case to the test function.

    vector_db_service is session-scoped (real ChromaDB + embedding model,
    expensive to set up), so the collection is cleared before each dataset is
    loaded. Otherwise documents from one test file's dataset stay in the
    collection and can outrank another file's expected results.
    """
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

def record_custom_retrieval_result(
    *, recorder, file_name, case_name, metric_name, score, threshold, passed, query,
    retrieved_contexts=None, expected_ids=None, retrieved_ids=None, filter=None, metadata=None,
):
    """Persists one Custom Retrieval check as an EvaluationResult.

    Score/threshold are recorded exactly as computed by the caller, never
    renormalized: boolean checks use 1.0/0.0 with threshold=None; raw-distance
    checks keep the actual distance and its native (lower-is-better)
    comparison direction. distance_stratification's two-sided band
    (min_dist/max_dist) doesn't fit the single `threshold` field, so its
    upper bound goes there and the full band goes in `metadata`.

    No Assistant model or Judge is involved in Retrieval, so
    assistant_model/judge_model/embedding_model are always None.
    """
    case_id = f"{file_name}::{case_name}"
    recorder.record(
        framework="custom",
        phase="retrieval",
        suite=file_name,
        case_id=case_id,
        case_name=case_name,
        metric=metric_name,
        quality_dimension=get_quality_dimension("custom", metric_name),
        query=query,
        threshold=threshold,
        status=Status.PASS if passed else Status.FAIL,
        score=score,
        metadata=metadata or {},
        retrieved_contexts=retrieved_contexts,
        expected_ids=expected_ids,
        retrieved_ids=retrieved_ids,
        filter=filter,
        judge_reason=None,
        assistant_model=None,
        judge_model=None,
        embedding_model=None,
    )
    write_latest(EVENTS_PATH, LATEST_PATH)
