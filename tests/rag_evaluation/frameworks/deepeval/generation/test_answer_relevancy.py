import pytest
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

from tests.rag_evaluation.frameworks.deepeval.generation.conftest import (
    get_generation_deepeval_cases,
    get_or_generate_response,
    record_generation_result,
)

pytestmark = pytest.mark.live


@pytest.mark.parametrize(
    "case", get_generation_deepeval_cases("answer_relevancy"), ids=lambda c: c["case_id"]
)
def test_answer_relevancy(assistant_llm_service, deepeval_judge_model, assistant_response_cache, recorder, case):
    """
    AnswerRelevancy: does the Assistant's real response actually address the
    user's question? Reference-free and context-free -- AnswerRelevancyMetric's
    required params are only input/actual_output (confirmed from source),
    so no retrieval_context/expected_output is needed or set.

    Applicability deliberately runs on all 21 cases with zero skips, NOT a
    copy of RAGAS AnswerRelevancy's skip list. Two reasons: (1) nothing
    required is ever missing -- every case has input and actual_output, so
    there is no structural reason to exclude any of them; (2) "does this
    answer address the question" is a meaningful, distinct axis even for
    refusal/absence-of-information and persona/tone cases -- it is exactly
    the axis RAGAS's own AnswerRelevancy got wrong on at least one case in
    this dataset (the parrot-exclusion case in llm_correctness.json, where a
    correct, highly-similar denial was hard-zeroed by RAGAS's noncommittal
    classifier). Deliberately including the same refusal/absence-of-info
    cases here, unfiltered, is what makes that RAGAS-vs-DeepEval comparison
    possible on identical cases instead of a RAGAS-filtered subset.
    DeepEval's algorithm has no equivalent explicit noncommittal classifier
    (confirmed in the DeepEval architecture audit) -- whether it reproduces
    or avoids RAGAS's false-zero is exactly what running it here measures.
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

    min_score = case["evaluation"]["deepeval"]["answer_relevancy"].get("min_score")
    if min_score is not None:
        assert metric.score >= min_score, (
            f"'{case['case_id']}': AnswerRelevancy {metric.score:.3f} below minimum {min_score:.3f}"
        )
