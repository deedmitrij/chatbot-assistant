import pytest
from deepeval.metrics import ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase

from tests.rag_evaluation.frameworks.deepeval.retrieval.conftest import (
    get_retrieval_deepeval_cases,
    search_retrieval,
    record_retrieval_result,
)

pytestmark = pytest.mark.live


@pytest.mark.parametrize(
    "test_case", get_retrieval_deepeval_cases(), ids=lambda item: item[1]["name"], indirect=True
)
def test_contextual_relevancy(vector_db_service, deepeval_judge_model, recorder, test_case):
    """
    ContextualRelevancy: are the chunks the real retrieval path returns for
    this query actually relevant to it? Query + retrieved context only -- no
    generated Assistant response, no reference answer needed. Confirmed by
    reading the installed metric's source (see the DeepEval architecture
    audit): actual_output is not in ContextualRelevancyMetric's required
    params and is never referenced in its scoring, so this runs on the same
    case population as RAGAS's ContextRelevance without ever calling the
    Assistant -- a DeepEval-native, different-algorithm counterpart to the
    same retrieval_relevance quality dimension.
    """
    # Retrieval preparation
    retrieved_contexts, retrieved_ids = search_retrieval(vector_db_service, test_case)

    # DeepEval evaluation
    llm_test_case = LLMTestCase(
        input=test_case["query"],
        retrieval_context=retrieved_contexts,
    )
    metric = ContextualRelevancyMetric(model=deepeval_judge_model, include_reason=True, async_mode=False)
    metric.measure(llm_test_case)

    # Reporting
    record_retrieval_result(
        recorder=recorder,
        test_case=test_case,
        metric_name="contextual_relevancy",
        score=metric.score,
        reason=metric.reason,
        retrieved_contexts=retrieved_contexts,
        retrieved_ids=retrieved_ids,
    )

    # Test validation
    print(f"\nContextualRelevancy[{test_case['name']}] = {metric.score}")

    assert isinstance(metric.score, (int, float)), (
        f"'{test_case['name']}': expected a numeric ContextualRelevancy score, got {metric.score!r}"
    )
    assert 0.0 <= metric.score <= 1.0, (
        f"'{test_case['name']}': ContextualRelevancy score {metric.score} out of [0, 1] range"
    )

    min_score = test_case["evaluation"]["deepeval"]["contextual_relevancy"].get("min_score")
    if min_score is not None:
        assert metric.score >= min_score, (
            f"'{test_case['name']}': ContextualRelevancy {metric.score} regressed below min_score {min_score}"
        )
