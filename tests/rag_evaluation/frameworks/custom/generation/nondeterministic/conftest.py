import pytest
from config import CHAT_MODEL, JUDGE_MODEL
from backend.constants import LLMRole
from backend.services.llm.llm_service import LLMService
from tests.rag_evaluation.reporting.paths import EVENTS_PATH, LATEST_PATH
from tests.rag_evaluation.reporting.aggregate import write_latest
from tests.rag_evaluation.reporting.dimensions import get_quality_dimension
from tests.rag_evaluation.reporting.schema import Status


@pytest.fixture(scope="session")
def llm_as_a_judge():
    """LLM service acting as the Judge. Session-scoped: a stateless config
    wrapper, safe to reuse across every evaluation."""
    return LLMService(role=LLMRole.JUDGE)

def record_custom_generation_result(
    *, recorder, file_name, test_case, passed,
    assistant_answer, actual_confidence, judge_reason=None, judge_model=None, metadata=None,
):
    """Persists the one primary EvaluationResult for an executed Custom
    Generation case.

    The metric is the suite's own theme (the Golden Dataset filename stem),
    not the internal guardrail name -- callers report a single outcome per
    case: FAILED at whichever guardrail or Judge check failed first (with
    diagnostics in `metadata`, e.g. failure_stage/missing_facts), or PASSED
    once every guardrail and the Judge succeeded. judge_model/judge_reason
    are only set by the caller on the Judge-verdict outcome -- guardrail
    failures never call the Judge.

    Custom's checks are boolean/pass-fail by design, not a numeric metric
    score, so score is 1.0/0.0 and threshold stays None.
    """
    metric_name = file_name.rsplit(".", 1)[0]
    case_id = f"{file_name}::{test_case['name']}"
    recorder.record(
        framework="custom",
        phase="generation",
        suite=file_name,
        case_id=case_id,
        case_name=test_case["name"],
        metric=metric_name,
        quality_dimension=get_quality_dimension("custom", metric_name),
        query=test_case["query"],
        threshold=None,
        status=Status.PASS if passed else Status.FAIL,
        score=1.0 if passed else 0.0,
        metadata=metadata or {},
        context=test_case["context"],
        assistant_answer=assistant_answer,
        reference_answer=test_case.get("reference_answer"),
        expected_confidence=test_case.get("expected_confidence"),
        actual_confidence=actual_confidence,
        judge_reason=judge_reason,
        assistant_model=CHAT_MODEL,
        judge_model=judge_model,
        embedding_model=None,
    )
    write_latest(EVENTS_PATH, LATEST_PATH)
