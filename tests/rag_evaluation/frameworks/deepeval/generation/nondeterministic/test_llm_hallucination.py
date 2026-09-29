import pytest
from deepeval.metrics import HallucinationMetric
from deepeval.test_case import LLMTestCase

from tests.rag_evaluation.frameworks.helpers import get_all_generation_test_cases_from_file, get_or_generate_response
from tests.rag_evaluation.frameworks.deepeval.generation.conftest import (
    record_generation_result,
    resolve_min_score,
)

pytestmark = pytest.mark.live


@pytest.mark.parametrize(
    "case", get_all_generation_test_cases_from_file("llm_hallucination.json"), ids=lambda c: c["case_id"]
)
def test_hallucination(assistant_llm_service, deepeval_judge_model, assistant_response_cache, recorder, case):
    """
    Hallucination: for each item in `context` (not `retrieval_context` --
    HallucinationMetric's required params are input/actual_output/context),
    does the Assistant's real response contradict it? Custom has no
    equivalent deterministic check; DeepEval is the only framework
    exercising this dimension. Reads llm_hallucination.json only.

    SCORE DIRECTION -- confirmed from the installed source (the Judge prompt
    literally asks "does the actual output AGREE with context?", scored as
    (# agreeing context items) / (# context items)) and from DeepEval's own
    docs: HIGHER IS BETTER, same direction as Faithfulness -- 1.0 means no
    contradictions found, 0.0 means every context item is contradicted.
    """
    # Assistant response preparation
    assistant_result = get_or_generate_response(assistant_response_cache, assistant_llm_service, case)
    response = assistant_result["answer"]

    # DeepEval evaluation
    llm_test_case = LLMTestCase(
        input=case["query"],
        actual_output=response,
        context=case["context"],
    )
    metric = HallucinationMetric(model=deepeval_judge_model, include_reason=True, async_mode=False)
    metric.measure(llm_test_case)

    # Reporting
    record_generation_result(
        recorder=recorder,
        case=case,
        metric_name="hallucination",
        score=metric.score,
        reason=metric.reason,
        assistant_result=assistant_result,
    )

    # Test validation
    print(f"\nHallucination[{case['case_id']}] = {metric.score}")

    assert isinstance(metric.score, (int, float)), (
        f"'{case['case_id']}': expected a numeric Hallucination score, got {metric.score!r}"
    )
    assert 0.0 <= metric.score <= 1.0, (
        f"'{case['case_id']}': Hallucination score {metric.score} out of [0, 1] range"
    )

    min_score = resolve_min_score(case["evaluation"]["deepeval"], "hallucination")
    if min_score is not None:
        assert metric.score >= min_score, (
            f"'{case['case_id']}': Hallucination {metric.score:.3f} below minimum {min_score:.3f}"
        )
