import pytest
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

from tests.rag_evaluation.frameworks.helpers import get_all_generation_test_cases_from_file, get_or_generate_response
from tests.rag_evaluation.frameworks.deepeval.generation.conftest import (
    record_generation_result,
    resolve_min_score,
)

pytestmark = pytest.mark.live


@pytest.mark.parametrize(
    "case", get_all_generation_test_cases_from_file("llm_relevancy.json"), ids=lambda c: c["case_id"]
)
def test_answer_relevancy(assistant_llm_service, deepeval_judge_model, assistant_response_cache, recorder, case):
    """
    AnswerRelevancy: does the Assistant's real response actually address the
    user's question? Reference-free and context-free -- AnswerRelevancyMetric's
    required params are only input/actual_output. DeepEval-native counterpart
    to RAGAS's AnswerRelevancy -- same relevancy quality dimension, no
    equivalent noncommittal-answer classifier. Reads llm_relevancy.json only.
    """
    # Assistant response preparation
    assistant_result = get_or_generate_response(assistant_response_cache, assistant_llm_service, case)
    response = assistant_result["answer"]

    # DeepEval evaluation
    llm_test_case = LLMTestCase(
        input=case["query"],
        actual_output=response,
    )
    metric = AnswerRelevancyMetric(model=deepeval_judge_model, include_reason=True, async_mode=False)
    metric.measure(llm_test_case)

    # Reporting
    record_generation_result(
        recorder=recorder,
        case=case,
        metric_name="answer_relevancy",
        score=metric.score,
        reason=metric.reason,
        assistant_result=assistant_result,
    )

    # Test validation
    print(f"\nAnswerRelevancy[{case['case_id']}] = {metric.score}")

    assert isinstance(metric.score, (int, float)), (
        f"'{case['case_id']}': expected a numeric AnswerRelevancy score, got {metric.score!r}"
    )
    assert 0.0 <= metric.score <= 1.0, (
        f"'{case['case_id']}': AnswerRelevancy score {metric.score} out of [0, 1] range"
    )

    min_score = resolve_min_score(case["evaluation"]["deepeval"], "answer_relevancy")
    if min_score is not None:
        assert metric.score >= min_score, (
            f"'{case['case_id']}': AnswerRelevancy {metric.score:.3f} below minimum {min_score:.3f}"
        )
