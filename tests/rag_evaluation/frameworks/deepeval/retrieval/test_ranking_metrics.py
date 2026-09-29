import pytest
from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric
from deepeval.test_case import LLMTestCase

from tests.rag_evaluation.frameworks.helpers import get_all_retrieval_test_cases_from_file
from tests.rag_evaluation.frameworks.deepeval.retrieval.conftest import (
    search_retrieval,
    record_retrieval_result,
    resolve_min_score,
)

pytestmark = pytest.mark.live

# ContextualPrecision/ContextualRecall both need expected_output and judge
# each retrieved chunk's support for the *whole* reference answer, so
# composite-reference cases (a reference synthesizing facts from more than
# one expected document) are structurally ineligible: no single chunk can
# support a multi-fact reference in full, so these metrics would score them
# near-zero regardless of actual retrieval quality. Those cases stay fully
# covered by Custom's ranking metrics.
RANKING_CASES = [
    case for case in get_all_retrieval_test_cases_from_file("ranking_metrics.json")
    if not case.get("evaluation", {}).get("reference_answer_is_composite", False)
]


@pytest.mark.parametrize("test_case", RANKING_CASES, ids=lambda c: c["name"], indirect=True)
def test_contextual_precision(vector_db_service, deepeval_judge_model, recorder, test_case):
    """
    ContextualPrecision: of what we retrieved, how much actually supports
    producing the reference answer, and is it ranked ahead of the noise?
    DeepEval-native counterpart to RAGAS's ContextPrecisionWithReference --
    same retrieval_precision quality dimension, different algorithm. Reads
    ranking_metrics.json only, excluding its composite-reference cases (see
    RANKING_CASES above).
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

    min_score = resolve_min_score(test_case["evaluation"]["deepeval"], "contextual_precision")
    if min_score is not None:
        assert metric.score >= min_score, (
            f"'{test_case['name']}': ContextualPrecision {metric.score} regressed below min_score {min_score}"
        )


@pytest.mark.parametrize("test_case", RANKING_CASES, ids=lambda c: c["name"], indirect=True)
def test_contextual_recall(vector_db_service, deepeval_judge_model, recorder, test_case):
    """
    ContextualRecall: of what the reference answer needs, how much did we
    actually retrieve? DeepEval-native counterpart to RAGAS's ContextRecall
    -- same retrieval_recall quality dimension, different algorithm (extracts
    statements from expected_output, then attributes each to
    retrieval_context, rather than RAGAS's own claim-NLI approach). Run on
    the same ranking_metrics.json case subset as ContextualPrecision, for
    the same reason (see RANKING_CASES above).
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

    min_score = resolve_min_score(test_case["evaluation"]["deepeval"], "contextual_recall")
    if min_score is not None:
        assert metric.score >= min_score, (
            f"'{test_case['name']}': ContextualRecall {metric.score} regressed below min_score {min_score}"
        )
