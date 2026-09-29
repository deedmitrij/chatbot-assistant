import pytest
from ragas.metrics.collections import ContextRelevance

from tests.rag_evaluation.frameworks.helpers import get_all_retrieval_test_cases_from_file
from tests.rag_evaluation.frameworks.ragas.retrieval.conftest import (
    search_retrieval,
    record_retrieval_result,
)

pytestmark = pytest.mark.live


@pytest.mark.parametrize(
    "test_case", get_all_retrieval_test_cases_from_file("top_k_retrieval.json"), ids=lambda c: c["name"], indirect=True
)
def test_context_relevance(vector_db_service, ragas_judge_llm, recorder, test_case):
    """
    Context Relevance: are the chunks the real retrieval path returns for this
    query actually relevant to it? Query + retrieved context only — no
    generated Assistant response, no reference answer needed. RAGAS-native
    counterpart to Custom's HitRate@K. Reads top_k_retrieval.json only.
    """
    # Retrieval data preparation
    retrieved_contexts, retrieved_ids = search_retrieval(vector_db_service, test_case)

    # RAGAS evaluation
    result = ContextRelevance(llm=ragas_judge_llm).score(
        user_input=test_case["query"], retrieved_contexts=retrieved_contexts
    )

    # Reporting
    record_retrieval_result(
        recorder=recorder,
        test_case=test_case,
        metric_name="context_relevance",
        result=result,
        retrieved_contexts=retrieved_contexts,
        retrieved_ids=retrieved_ids,
    )

    # Test validation
    print(f"\nContextRelevance[{test_case['name']}] = {result.value}")

    assert isinstance(result.value, (int, float)), (
        f"'{test_case['name']}': expected a numeric ContextRelevance score, got {result.value!r}"
    )
    assert 0.0 <= result.value <= 1.0, (
        f"'{test_case['name']}': ContextRelevance score {result.value} out of [0, 1] range"
    )

    min_score = test_case["evaluation"]["ragas"]["min_score"]
    assert result.value >= min_score, (
        f"'{test_case['name']}': ContextRelevance {result.value} regressed below min_score {min_score}"
    )
