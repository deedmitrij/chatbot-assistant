import pytest
from deepeval.metrics import HallucinationMetric
from deepeval.test_case import LLMTestCase

from tests.rag_evaluation.frameworks.deepeval.generation.conftest import (
    get_generation_deepeval_cases,
    get_or_generate_response,
    record_generation_result,
)

pytestmark = pytest.mark.live


@pytest.mark.parametrize(
    "case", get_generation_deepeval_cases("hallucination"), ids=lambda c: c["case_id"]
)
def test_hallucination(assistant_llm_service, deepeval_judge_model, assistant_response_cache, recorder, case):
    """
    Hallucination: for each item in `context` (not `retrieval_context` --
    HallucinationMetric's required params are input/actual_output/context),
    does the Assistant's real response contradict it?

    SCORE DIRECTION -- confirmed from the installed source (the Judge prompt
    literally asks "does the actual output AGREE with context?", scored as
    (# agreeing context items) / (# context items)) and from DeepEval's own
    docs: HIGHER IS BETTER, same direction as Faithfulness -- 1.0 means no
    contradictions found, 0.0 means every context item is contradicted. This
    is the opposite of what the metric's name might suggest (a raw
    "contamination rate"), so record_generation_result()'s existing
    min_score convention (status=PASS when score >= min_score) is already
    correct here with no inversion or special-casing.

    Applicability deliberately does NOT copy RAGAS Faithfulness's skip list:
    Hallucination only flags outright contradiction ("You should FORGIVE
    cases where actual output is lacking in detail... ONLY provide a 'no'
    answer if IT IS A CONTRADICTION" -- from the installed prompt), so a
    correct "this detail isn't specified" answer about an attribute the
    context is merely silent on does NOT contradict that context and scores
    well. This is why all 4 llm_hallucination.json cases are applicable here,
    including "Pool Temperature", which RAGAS Faithfulness itself skips for
    the opposite reason (its claim-decomposition approach needs the claim to
    be entailed, not merely non-contradicted). See
    evaluation.deepeval.hallucination.skip_reason on the 3 cases that ARE
    skipped for the specific, confirmed reason in each.
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

    min_score = case["evaluation"]["deepeval"]["hallucination"].get("min_score")
    if min_score is not None:
        assert metric.score >= min_score, (
            f"'{case['case_id']}': Hallucination {metric.score:.3f} below minimum {min_score:.3f}"
        )
