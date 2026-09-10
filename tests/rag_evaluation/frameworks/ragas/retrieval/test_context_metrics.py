import pytest
from ragas.metrics.collections import ContextRelevance

from tests.rag_evaluation.frameworks.ragas.retrieval.conftest import get_retrieval_ragas_cases

pytestmark = pytest.mark.live

# ContextRecall was evaluated for this suite too, but RAGAS 0.4.3's
# ContextRecall.ascore() takes `reference` as a single ground-truth *answer*
# string, not reference context ids/text. Our golden retrieval data has no
# such answer, only expected_id(s). Synthesizing one by concatenating the
# expected documents' text would invent a semantic the metric was never
# designed to score, rather than reuse existing data as-is — so ContextRecall
# is deliberately deferred until a non-invented mapping is agreed, not
# implemented here.


@pytest.mark.parametrize(
    "test_case", get_retrieval_ragas_cases(), ids=lambda item: item[1]["name"], indirect=True
)
def test_context_relevance(vector_db_service, ragas_judge_llm, test_case):
    """
    Context Relevance: are the chunks the real retrieval path returns for this
    query actually relevant to it? Query + retrieved context only — no
    generated Assistant response, no reference answer — so this stays a pure
    retrieval-quality check, comparable in intent (not value) to Custom's own
    retrieval metrics without reusing their mechanism.
    """
    n_results = test_case.get("n_results", 3)
    search_result = vector_db_service.search(
        query_text=test_case["query"],
        n_results=n_results,
        where_filter=test_case.get("filter"),
    )
    retrieved_contexts = search_result["documents"][0]

    metric = ContextRelevance(llm=ragas_judge_llm)
    result = metric.score(user_input=test_case["query"], retrieved_contexts=retrieved_contexts)

    print(f"\nContextRelevance[{test_case['name']}] = {result.value}")

    assert isinstance(result.value, (int, float)), (
        f"'{test_case['name']}': expected a numeric ContextRelevance score, got {result.value!r}"
    )
    assert 0.0 <= result.value <= 1.0, (
        f"'{test_case['name']}': ContextRelevance score {result.value} out of [0, 1] range"
    )
