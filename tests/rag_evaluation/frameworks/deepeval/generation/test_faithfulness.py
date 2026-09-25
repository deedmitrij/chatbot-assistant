import pytest
from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase

from tests.rag_evaluation.frameworks.deepeval.generation.conftest import (
    get_generation_deepeval_cases,
    get_or_generate_response,
    record_generation_result,
)

pytestmark = pytest.mark.live


@pytest.mark.parametrize(
    "case", get_generation_deepeval_cases("faithfulness"), ids=lambda c: c["case_id"]
)
def test_faithfulness(assistant_llm_service, deepeval_judge_model, assistant_response_cache, recorder, case):
    """
    Faithfulness: is the Assistant's real response supported by the same
    context it was given? DeepEval-native counterpart to RAGAS's Faithfulness
    -- same groundedness quality dimension, different algorithm (extracts
    'truths' from retrieval_context and 'claims' from actual_output, then
    classifies each claim's verdict against the truths, rather than RAGAS's
    own claim-decomposition + NLI verification).
    """
    # Assistant response preparation
    assistant_result = get_or_generate_response(assistant_response_cache, assistant_llm_service, case)
    response = assistant_result["answer"]

    # DeepEval evaluation
    llm_test_case = LLMTestCase(
        input=case["query"],
        actual_output=response,
        retrieval_context=case["context"],
    )
    metric = FaithfulnessMetric(model=deepeval_judge_model, include_reason=True, async_mode=False)
    metric.measure(llm_test_case)

    # Reporting
    record_generation_result(
        recorder=recorder,
        case=case,
        metric_name="faithfulness",
        score=metric.score,
        reason=metric.reason,
        assistant_result=assistant_result,
    )

    # Test validation
    print(f"\nFaithfulness[{case['case_id']}] = {metric.score}")

    assert isinstance(metric.score, (int, float)), (
        f"'{case['case_id']}': expected a numeric Faithfulness score, got {metric.score!r}"
    )
    assert 0.0 <= metric.score <= 1.0, (
        f"'{case['case_id']}': Faithfulness score {metric.score} out of [0, 1] range"
    )

    min_score = case["evaluation"]["deepeval"]["faithfulness"].get("min_score")
    if min_score is not None:
        assert metric.score >= min_score, (
            f"'{case['case_id']}': Faithfulness {metric.score:.3f} below minimum {min_score:.3f}"
        )
