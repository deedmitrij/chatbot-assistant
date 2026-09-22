import pytest
from ragas.metrics.collections import AnswerRelevancy

from tests.rag_evaluation.frameworks.ragas.generation.conftest import (
    ANSWER_RELEVANCY_STRICTNESS,
    get_generation_ragas_cases,
    get_or_generate_response,
    record_generation_result,
)
from config import EMBEDDING_MODEL

pytestmark = pytest.mark.live


@pytest.mark.parametrize(
    "case", get_generation_ragas_cases("answer_relevancy"), ids=lambda c: c["case_id"]
)
def test_answer_relevancy(
    assistant_llm_service, ragas_judge_llm, ragas_embeddings, assistant_response_cache, recorder, case
):
    """
    AnswerRelevancy: does the Assistant's real response actually address the
    user's question? Reference-free and context-free — self-consistency
    check via paraphrase-question generation and cosine similarity to the
    original query, which also detects evasive/noncommittal answers.
    """
    assistant_result = get_or_generate_response(assistant_response_cache, assistant_llm_service, case)
    response = assistant_result["answer"]

    # RAGAS evaluation
    metric = AnswerRelevancy(
        llm=ragas_judge_llm, embeddings=ragas_embeddings, strictness=ANSWER_RELEVANCY_STRICTNESS
    )
    result = metric.score(user_input=case["query"], response=response)

    # Reporting
    record_generation_result(
        recorder=recorder,
        case=case,
        metric_name="answer_relevancy",
        result=result,
        assistant_result=assistant_result,
        embedding_model=EMBEDDING_MODEL,
    )

    # Test validation
    assert isinstance(result.value, (int, float)), (
        f"'{case['case_id']}': expected a numeric AnswerRelevancy score, got {result.value!r}"
    )
    assert 0.0 <= result.value <= 1.0, (
        f"'{case['case_id']}': AnswerRelevancy score {result.value} out of [0, 1] range"
    )

    min_score = case["evaluation"]["ragas"]["answer_relevancy"]["min_score"]
    assert result.value >= min_score, (
        f"'{case['case_id']}': AnswerRelevancy score {result.value:.3f} "
        f"below minimum {min_score:.3f}"
    )
