import pytest
from deepeval.metrics import ContextualPrecisionMetric
from deepeval.test_case import LLMTestCase

from tests.rag_evaluation.frameworks.deepeval.retrieval.conftest import (
    get_single_reference_deepeval_cases,
    search_retrieval,
    record_retrieval_result,
)

pytestmark = pytest.mark.live


@pytest.mark.parametrize(
    "test_case", get_single_reference_deepeval_cases(), ids=lambda item: item[1]["name"], indirect=True
)
def test_contextual_precision(vector_db_service, deepeval_judge_model, recorder, test_case):
    """
    ContextualPrecision: of what we retrieved, how much actually supports
    producing the reference answer, and is it ranked ahead of the noise?
    DeepEval-native counterpart to RAGAS's ContextPrecisionWithReference --
    same retrieval_precision quality dimension, different algorithm. Only
    runs on cases whose reference_answer is a single, naturally supported
    factual target (see get_single_reference_deepeval_cases); a composite
    reference is excluded rather than scored near-zero regardless of actual
    retrieval quality.
    """
    # Retrieval preparation
    retrieved_contexts, retrieved_ids = search_retrieval(vector_db_service, test_case)

    # DeepEval evaluation
    llm_test_case = LLMTestCase(
        input=test_case["query"],
        retrieval_context=retrieved_contexts,
        expected_output=test_case["reference_answer"],
    )
    metric = ContextualPrecisionMetric(model=deepeval_judge_model, include_reason=True, async_mode=False)
    metric.measure(llm_test_case)

    # Reporting
    record_retrieval_result(
        recorder=recorder,
        test_case=test_case,
        metric_name="contextual_precision",
        score=metric.score,
        reason=metric.reason,
        retrieved_contexts=retrieved_contexts,
        retrieved_ids=retrieved_ids,
    )

    # Test validation
    print(f"\nContextualPrecision[{test_case['name']}] = {metric.score}")

    assert isinstance(metric.score, (int, float)), (
        f"'{test_case['name']}': expected a numeric ContextualPrecision score, got {metric.score!r}"
    )
    assert 0.0 <= metric.score <= 1.0, (
        f"'{test_case['name']}': ContextualPrecision score {metric.score} out of [0, 1] range"
    )

    min_score = test_case["evaluation"]["deepeval"]["contextual_precision"].get("min_score")
    if min_score is not None:
        assert metric.score >= min_score, (
            f"'{test_case['name']}': ContextualPrecision {metric.score} regressed below min_score {min_score}"
        )
