import pytest
from ragas.metrics.collections import FactualCorrectness

from tests.rag_evaluation.frameworks.ragas.generation.conftest import (
    get_generation_ragas_cases,
    get_or_generate_response,
    record_generation_result,
)

pytestmark = pytest.mark.live


@pytest.mark.parametrize(
    "case", get_generation_ragas_cases("factual_correctness"), ids=lambda c: c["case_id"]
)
def test_factual_correctness(assistant_llm_service, ragas_judge_llm, assistant_response_cache, recorder, case):
    """
    FactualCorrectness: does the Assistant's real response factually agree
    with the human-authored reference_answer? Bidirectional claim
    decomposition + NLI (default f1 mode) — no query context, no
    embeddings, purely response-vs-reference claim overlap.
    """
    assistant_result = get_or_generate_response(assistant_response_cache, assistant_llm_service, case)
    response = assistant_result["answer"]
    reference = case["reference_answer"]

    # RAGAS evaluation
    metric = FactualCorrectness(llm=ragas_judge_llm)
    result = metric.score(response=response, reference=reference)

    # Reporting
    record_generation_result(
        recorder=recorder,
        case=case,
        metric_name="factual_correctness",
        result=result,
        assistant_result=assistant_result,
    )

    # Test validation
    assert isinstance(result.value, (int, float)), (
        f"'{case['case_id']}': expected a numeric FactualCorrectness score, got {result.value!r}"
    )
    assert 0.0 <= result.value <= 1.0, (
        f"'{case['case_id']}': FactualCorrectness score {result.value} out of [0, 1] range"
    )

    min_score = case["evaluation"]["ragas"]["factual_correctness"]["min_score"]
    assert result.value >= min_score, (
        f"'{case['case_id']}': FactualCorrectness score {result.value:.3f} "
        f"below minimum {min_score:.3f}"
    )
