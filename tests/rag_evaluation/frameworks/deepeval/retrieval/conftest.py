import json
import os
import pytest
from deepeval.models import OllamaModel
from config import PROJECT_ROOT, RAG_EVALUATION_DATA_DIR, JUDGE_MODEL
from backend.services.vector_db_service import VectorDBService
from tests.rag_evaluation.reporting.recorder import Recorder
from tests.rag_evaluation.reporting.aggregate import write_latest
from tests.rag_evaluation.reporting.dimensions import get_quality_dimension
from tests.rag_evaluation.reporting.schema import Status

EVENTS_PATH = PROJECT_ROOT / "reports" / "results" / "events.jsonl"
LATEST_PATH = PROJECT_ROOT / "reports" / "results" / "latest.json"

# The default 88.5s per-attempt timeout is too short for qwen3:8b's
# thinking-mode latency: the native OllamaModel integration has no way to
# disable thinking mode (unlike LLMService's extra_body={"reasoning_effort":
# "none"}), confirmed in the prior DeepEval Judge smoke test. Extending this
# here, not globally, keeps the override scoped to this framework's tests.
os.environ.setdefault("DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE", "240")

# Only the golden files that carry an expected_id/expected_ids (a golden
# document) are usable here -- same file set and same eligibility rule as
# RAGAS's ContextRelevance (see ../../ragas/retrieval/conftest.py), so
# ContextualRelevancyMetric runs on the identical case population.
RETRIEVAL_FILES_WITH_EXPECTED_DOCS = (
    "query_robustness.json",
    "top_k_retrieval.json",
    "metadata_filtering.json",
    "ranking_metrics.json",
)


@pytest.fixture(scope="session")
def vector_db_service():
    # Distinct collection name from Custom's "test_collection" and RAGAS's
    # "ragas_test_collection" so all three suites can run in the same pytest
    # session without interfering with each other.
    collection = "deepeval_test_collection"
    db = VectorDBService(collection=collection)
    yield db
    db.client.delete_collection(collection)


@pytest.fixture(scope="session")
def deepeval_judge_model():
    """Native DeepEval Ollama Judge, wired to the same local Judge model as
    the Custom/RAGAS frameworks (JUDGE_MODEL, qwen3:8b). Project-scoped
    construction -- no `deepeval set-ollama` global config."""
    return OllamaModel(
        model=JUDGE_MODEL,
        base_url="http://localhost:11434",
        temperature=0,
    )


def get_retrieval_deepeval_cases():
    """Utility to load DeepEval-eligible retrieval test cases from the shared
    golden data. Same (dataset, case) tuple shape as Custom's and RAGAS's own
    loaders. Used by ContextualRelevancyMetric, which needs no reference
    answer and no actual_output."""
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


def get_single_reference_deepeval_cases():
    """Subset of get_retrieval_deepeval_cases() whose reference_answer is a
    single, naturally supported factual target -- used by
    ContextualPrecisionMetric and ContextualRecallMetric, which both need
    expected_output and judge each retrieved chunk's support for the *whole*
    reference answer. Cases marked reference_answer_is_composite (a reference
    synthesizing facts from more than one expected document) are excluded:
    no single chunk can support a multi-fact reference in full, so these
    metrics would score them near-zero regardless of retrieval quality.
    Same exclusion rule as RAGAS's get_single_reference_ragas_cases (see
    ../../ragas/retrieval/conftest.py) -- those cases stay fully covered by
    ContextualRelevancyMetric and by Custom's ranking metrics."""
    return [
        (dataset, case)
        for dataset, case in get_retrieval_deepeval_cases()
        if not case.get("evaluation", {}).get("reference_answer_is_composite", False)
    ]


@pytest.fixture
def test_case(vector_db_service, request):
    """Load the case's dataset into the vector database, then hand back the
    case. Mirrors Custom's and RAGAS's own test_case fixture exactly (see
    ../../ragas/retrieval/conftest.py)."""
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
    """One Recorder per pytest session, persisting to the same events.jsonl
    the reporting subsystem's aggregator reads from -- same pattern as
    Custom's and RAGAS's own conftest.py files."""
    return Recorder(EVENTS_PATH)


def search_retrieval(vector_db_service, test_case):
    """One ChromaDB search per case, same query/top_k/filter/ordering as
    Custom and RAGAS. Returns (retrieved_contexts, retrieved_ids) from that
    single search result."""
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


def record_retrieval_result(*, recorder, test_case, metric_name, score, reason, retrieved_contexts, retrieved_ids):
    """Reporting-only: persists one EvaluationResult for an already-measured
    DeepEval metric's score/reason, then re-materializes latest.json. Mirrors
    RAGAS's record_retrieval_result (see ../../ragas/retrieval/conftest.py),
    adapted for DeepEval's plain (score, reason) attributes instead of a
    RAGAS MetricResult object -- never constructs or measures the metric.

    No Assistant model is involved in Retrieval at all (no actual_output on
    the LLMTestCase, no LLM generation step, only vector search + Judge
    scoring), so assistant_model and embedding_model are always None.
    """
    is_nan = score != score
    min_score = test_case["evaluation"]["deepeval"][metric_name].get("min_score")

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
