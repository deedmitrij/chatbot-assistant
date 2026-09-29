import pytest
from ragas.embeddings.huggingface_provider import HuggingFaceEmbeddings
from config import CHAT_MODEL, JUDGE_MODEL, EMBEDDING_MODEL
from tests.rag_evaluation.reporting.paths import EVENTS_PATH, LATEST_PATH
from tests.rag_evaluation.reporting.aggregate import write_latest
from tests.rag_evaluation.reporting.dimensions import get_quality_dimension
from tests.rag_evaluation.reporting.schema import Status
from tests.rag_evaluation.frameworks.ragas.conftest import resolve_min_score

# Project choice, not a reproduction of RAGAS's benchmark defaults: this is a
# local AI Quality Engineering lab comparing evaluation frameworks, not an
# attempt to match RAGAS's benchmark-grade settings. AnswerRelevancy's
# default strictness=3 means 3 Judge calls per case; strictness=1 keeps the
# metric's semantics (paraphrase generation + cosine similarity against the
# original query, noncommittal-answer detection) while cutting the full
# Generation baseline's Judge-call count roughly in half.
ANSWER_RELEVANCY_STRICTNESS = 1


@pytest.fixture(scope="session")
def ragas_embeddings():
    """Local, offline embeddings for AnswerRelevancy/AnswerCorrectness --
    Generation-only, run locally via sentence-transformers (no network
    dependency)."""
    return HuggingFaceEmbeddings(model=EMBEDDING_MODEL, use_api=False)


def record_generation_result(*, recorder, case, metric_name, result, assistant_result, embedding_model=None):
    """Persists one EvaluationResult for an already-computed RAGAS
    MetricResult. Never constructs a metric or calls .score() itself.

    Status: NaN -> N/A; finite value with min_score configured -> PASS/FAIL;
    finite value with no min_score yet -> BASELINE.
    """
    is_nan = result.value != result.value
    min_score = resolve_min_score(case["evaluation"]["ragas"], metric_name)

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
        phase="generation",
        suite=case["case_id"].split("::", 1)[0],
        case_id=case["case_id"],
        case_name=case["name"],
        metric=metric_name,
        quality_dimension=get_quality_dimension("ragas", metric_name),
        query=case["query"],
        threshold=threshold,
        status=status,
        score=score,
        context=case["context"],
        assistant_answer=assistant_result["answer"],
        reference_answer=case.get("reference_answer"),
        expected_confidence=case.get("expected_confidence"),
        actual_confidence=assistant_result["confidence"],
        judge_reason=getattr(result, "reason", None),
        assistant_model=CHAT_MODEL,
        judge_model=JUDGE_MODEL,
        embedding_model=embedding_model,
    )
    write_latest(EVENTS_PATH, LATEST_PATH)
