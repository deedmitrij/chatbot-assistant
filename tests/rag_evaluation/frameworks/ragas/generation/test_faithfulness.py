import pytest
from ragas.metrics.collections import Faithfulness

from tests.rag_evaluation.frameworks.ragas.generation.conftest import (
    get_generation_ragas_cases,
    get_or_generate_response,
    record_generation_result,
)

pytestmark = pytest.mark.live


@pytest.mark.parametrize(
    "case", get_generation_ragas_cases("faithfulness"), ids=lambda c: c["case_id"]
)
def test_faithfulness(assistant_llm_service, ragas_judge_llm, assistant_response_cache, recorder, case):
    """
    Faithfulness: is the Assistant's real response supported by the same
    context it was given? Reference-free — RAGAS's claim-decomposition +
    NLI verification, a different methodology from Custom's holistic judge
    call for the same dimension.

    Some refusal/no-claim answers could in principle decompose into zero
    checkable statements, which RAGAS reports as NaN rather than a 0-1
    score. We don't special-case that here: NaN fails the range assertion
    below like any other invalid score, rather than silently bypassing the
    quality gate.
    """
    assistant_result = get_or_generate_response(assistant_response_cache, assistant_llm_service, case)
    response = assistant_result["answer"]

    # RAGAS evaluation
    metric = Faithfulness(llm=ragas_judge_llm)
    result = metric.score(
        user_input=case["query"], response=response, retrieved_contexts=case["context"]
    )

    # Reporting
    record_generation_result(
        recorder=recorder,
        case=case,
        metric_name="faithfulness",
        result=result,
        assistant_result=assistant_result,
    )

    # Test validation
    assert isinstance(result.value, (int, float)), (
        f"'{case['case_id']}': expected a numeric Faithfulness score, got {result.value!r}"
    )
    assert 0.0 <= result.value <= 1.0, (
        f"'{case['case_id']}': Faithfulness score {result.value} out of [0, 1] range"
    )

    min_score = case["evaluation"]["ragas"]["faithfulness"]["min_score"]
    assert result.value >= min_score, (
        f"'{case['case_id']}': Faithfulness score {result.value:.3f} "
        f"below minimum {min_score:.3f}"
    )
