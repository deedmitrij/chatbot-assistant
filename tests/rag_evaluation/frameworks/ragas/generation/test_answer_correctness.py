import pytest
from ragas.metrics.collections import AnswerCorrectness

from tests.rag_evaluation.frameworks.ragas.generation.conftest import (
    get_generation_ragas_cases,
    get_or_generate_response,
    record_generation_result,
)
from config import EMBEDDING_MODEL

pytestmark = pytest.mark.live


@pytest.mark.parametrize(
    "case", get_generation_ragas_cases("answer_correctness"), ids=lambda c: c["case_id"]
)
def test_answer_correctness(
    assistant_llm_service, ragas_judge_llm, ragas_embeddings, assistant_response_cache, recorder, case
):
    """
    AnswerCorrectness: weighted combination of factuality (query-aware
    statement classification against the reference_answer) and semantic
    similarity (response vs reference embeddings). Kept alongside
    FactualCorrectness deliberately, to compare their behavior empirically
    across the 17 shared cases rather than assume redundancy.
    """
    assistant_result = get_or_generate_response(assistant_response_cache, assistant_llm_service, case)
    response = assistant_result["answer"]
    reference = case["reference_answer"]

    # RAGAS evaluation
    metric = AnswerCorrectness(llm=ragas_judge_llm, embeddings=ragas_embeddings)
    result = metric.score(user_input=case["query"], response=response, reference=reference)

    # Reporting
    record_generation_result(
        recorder=recorder,
        case=case,
        metric_name="answer_correctness",
        result=result,
        assistant_result=assistant_result,
        embedding_model=EMBEDDING_MODEL,
    )

    # Test validation
    assert isinstance(result.value, (int, float)), (
        f"'{case['case_id']}': expected a numeric AnswerCorrectness score, got {result.value!r}"
    )
    assert 0.0 <= result.value <= 1.0, (
        f"'{case['case_id']}': AnswerCorrectness score {result.value} out of [0, 1] range"
    )

    min_score = case["evaluation"]["ragas"]["answer_correctness"]["min_score"]
    assert result.value >= min_score, (
        f"'{case['case_id']}': AnswerCorrectness score {result.value:.3f} "
        f"below minimum {min_score:.3f}"
    )
