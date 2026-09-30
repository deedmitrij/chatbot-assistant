import typing as t
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


class RubricResult(t.NamedTuple):
    """Minimal (value, reason) shape so a normalized InstanceSpecificRubrics
    score can be handed to record_generation_result exactly like a real
    RAGAS MetricResult, without constructing one."""
    value: float
    reason: str


def build_instance_rubrics(criteria: str) -> dict:
    """Builds a 5-level InstanceSpecificRubrics severity rubric for one
    Golden Dataset case, using its own `criteria` text as the semantic
    target being graded -- the rubric only defines grade severity."""
    return {
        "score1_description": f"Completely fails or contradicts the evaluation criteria: {criteria}",
        "score2_description": f"Substantially fails to meet the evaluation criteria: {criteria}",
        "score3_description": f"Partially meets the evaluation criteria, with at least one meaningful issue: {criteria}",
        "score4_description": f"Mostly meets the evaluation criteria, with only minor issues: {criteria}",
        "score5_description": f"Fully meets the evaluation criteria, with no issues: {criteria}",
    }


def normalize_rubric_score(raw_value: float) -> float:
    """Maps InstanceSpecificRubrics' native 1-5 scale to the project's
    0.0-1.0 reporting scale."""
    return (raw_value - 1) / 4


def record_generation_result(
    *, recorder, case, metric_name, result, assistant_result, embedding_model=None, metadata=None,
):
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
        metadata=metadata or {},
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
