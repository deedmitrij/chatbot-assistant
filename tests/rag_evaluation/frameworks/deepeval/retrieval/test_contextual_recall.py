import pytest
from deepeval.metrics import ContextualRecallMetric
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
def test_contextual_recall(vector_db_service, deepeval_judge_model, recorder, test_case):
    """
    ContextualRecall: of what the reference answer needs, how much did we
    actually retrieve? DeepEval-native counterpart to RAGAS's ContextRecall
    -- same retrieval_recall quality dimension, different algorithm (extracts
    statements from expected_output, then attributes each to
    retrieval_context, rather than RAGAS's own claim-NLI approach). Run on
    the same single-reference case subset as ContextualPrecision, for the
    same reason (see get_single_reference_deepeval_cases).
    """
    # Retrieval preparation
    retrieved_contexts, retrieved_ids = search_retrieval(vector_db_service, test_case)

    # DeepEval evaluation
    llm_test_case = LLMTestCase(
        input=test_case["query"],
        retrieval_context=retrieved_contexts,
        expected_output=test_case["reference_answer"],
    )
    metric = ContextualRecallMetric(model=deepeval_judge_model, include_reason=True, async_mode=False)
    metric.measure(llm_test_case)

    # Reporting
    record_retrieval_result(
        recorder=recorder,
        test_case=test_case,
        metric_name="contextual_recall",
        score=metric.score,
        reason=metric.reason,
        retrieved_contexts=retrieved_contexts,
        retrieved_ids=retrieved_ids,
    )

    # Test validation
    print(f"\nContextualRecall[{test_case['name']}] = {metric.score}")

    assert isinstance(metric.score, (int, float)), (
        f"'{test_case['name']}': expected a numeric ContextualRecall score, got {metric.score!r}"
    )
    assert 0.0 <= metric.score <= 1.0, (
        f"'{test_case['name']}': ContextualRecall score {metric.score} out of [0, 1] range"
    )

    min_score = test_case["evaluation"]["deepeval"]["contextual_recall"].get("min_score")
    if min_score is not None:
        assert metric.score >= min_score, (
            f"'{test_case['name']}': ContextualRecall {metric.score} regressed below min_score {min_score}"
        )
