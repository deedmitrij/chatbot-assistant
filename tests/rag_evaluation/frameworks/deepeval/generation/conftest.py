from config import CHAT_MODEL, JUDGE_MODEL
from reporting.paths import EVENTS_PATH, LATEST_PATH
from reporting.aggregate import write_latest
from reporting.dimensions import get_quality_dimension
from reporting.schema import Status
from tests.rag_evaluation.frameworks.deepeval.conftest import resolve_min_score


def record_generation_result(*, recorder, case, metric_name, score, reason, assistant_result):
    """Persists one EvaluationResult for an already-measured DeepEval
    metric's score/reason. Never constructs or measures the metric itself.

    Status: NaN -> N/A; finite value with min_score configured -> PASS/FAIL;
    finite value with no min_score yet -> BASELINE.
    """
    is_nan = score != score
    min_score = resolve_min_score(case["evaluation"]["deepeval"], metric_name)

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
        phase="generation",
        suite=case["case_id"].split("::", 1)[0],
        case_id=case["case_id"],
        case_name=case["name"],
        metric=metric_name,
        quality_dimension=get_quality_dimension("deepeval", metric_name),
        query=case["query"],
        threshold=threshold,
        status=status,
        score=recorded_score,
        context=case["context"],
        assistant_answer=assistant_result["answer"],
        reference_answer=case.get("reference_answer"),
        expected_confidence=case.get("expected_confidence"),
        actual_confidence=assistant_result["confidence"],
        judge_reason=reason,
        assistant_model=CHAT_MODEL,
        judge_model=JUDGE_MODEL,
        embedding_model=None,
    )
    write_latest(EVENTS_PATH, LATEST_PATH)
