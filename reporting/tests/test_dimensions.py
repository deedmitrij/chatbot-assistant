import pytest

from reporting.dimensions import get_quality_dimension


@pytest.mark.parametrize(
    "framework,metric,expected",
    [
        ("ragas", "faithfulness", "groundedness"),
        ("ragas", "answer_relevancy", "relevancy"),
        ("ragas", "factual_correctness", "correctness"),
        ("ragas", "answer_correctness", "correctness"),
        ("ragas", "hallucination", "hallucination"),
        ("ragas", "negative_constraint", "refusal"),
        ("ragas", "brand_consistency", "persona"),
        ("ragas", "context_relevance", "retrieval_relevance"),
        ("ragas", "context_precision_with_reference", "retrieval_precision"),
        ("ragas", "context_recall", "retrieval_recall"),
        ("custom", "llm_faithfulness", "groundedness"),
        ("custom", "llm_correctness", "correctness"),
        ("custom", "llm_relevancy", "relevancy"),
        ("custom", "llm_hallucination", "hallucination"),
        ("custom", "llm_negative_constraint", "refusal"),
        ("custom", "llm_brand_consistency", "persona"),
        # distance_stratification is deliberately its own dimension, not
        # left unmapped and not folded into "ranking".
        ("custom", "distance_stratification", "decision_thresholding"),
        ("deepeval", "contextual_relevancy", "retrieval_relevance"),
        ("deepeval", "contextual_precision", "retrieval_precision"),
        ("deepeval", "contextual_recall", "retrieval_recall"),
        ("deepeval", "faithfulness", "groundedness"),
        ("deepeval", "hallucination", "hallucination"),
        ("deepeval", "answer_relevancy", "relevancy"),
        ("deepeval", "geval_factual_correctness", "correctness"),
        ("deepeval", "geval_answer_correctness", "correctness"),
        ("deepeval", "geval_negative_constraint", "refusal"),
        ("deepeval", "geval_brand_consistency", "persona"),
    ],
)
def test_known_mappings(framework, metric, expected):
    assert get_quality_dimension(framework, metric) == expected


def test_unknown_metric_returns_none_not_a_guess():
    assert get_quality_dimension("ragas", "some_future_metric_not_yet_mapped") is None
