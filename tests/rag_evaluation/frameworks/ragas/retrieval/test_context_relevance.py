import pytest
from ragas.metrics.collections import ContextRelevance

from tests.rag_evaluation.frameworks.ragas.retrieval.conftest import (
    get_retrieval_ragas_cases,
    _retrieved_contexts,
)

pytestmark = pytest.mark.live


@pytest.mark.parametrize(
    "test_case", get_retrieval_ragas_cases(), ids=lambda item: item[1]["name"], indirect=True
)
def test_context_relevance(vector_db_service, ragas_judge_llm, test_case):
    """
    Context Relevance: are the chunks the real retrieval path returns for this
    query actually relevant to it? Query + retrieved context only — no
    generated Assistant response, no reference answer needed — so this runs
    for every RAGAS-eligible case, including the multi-relevant/composite-
    answer ones that ContextPrecisionWithReference/ContextRecall skip.
    """
    retrieved_contexts = _retrieved_contexts(vector_db_service, test_case)

    result = ContextRelevance(llm=ragas_judge_llm).score(
        user_input=test_case["query"], retrieved_contexts=retrieved_contexts
    )

    print(f"\nContextRelevance[{test_case['name']}] = {result.value}")

    assert isinstance(result.value, (int, float)), (
        f"'{test_case['name']}': expected a numeric ContextRelevance score, got {result.value!r}"
    )
    assert 0.0 <= result.value <= 1.0, (
        f"'{test_case['name']}': ContextRelevance score {result.value} out of [0, 1] range"
    )

    min_score = test_case["evaluation"]["ragas"]["context_relevance"]["min_score"]
    assert result.value >= min_score, (
        f"'{test_case['name']}': ContextRelevance {result.value} regressed below min_score {min_score}"
    )
