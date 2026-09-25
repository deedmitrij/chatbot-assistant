"""Maps each framework's native metric name to a shared quality-dimension
vocabulary, so results from different frameworks can be compared without
forcing their raw metrics to look identical.

Deliberately explicit, not inferred from string-matching metric names: a
mapping only exists here if it's semantically honest. Every currently-shipped
metric has an entry — there is no unmapped case left as of this table, but
get_quality_dimension() still returns None for anything absent rather than
guessing, so a future metric added without an entry here fails visibly
(shows up with no dimension in reports) instead of silently mis-mapping.
"""

QUALITY_DIMENSIONS = {
    "ragas": {
        "context_relevance": "retrieval_relevance",
        "context_precision_with_reference": "retrieval_precision",
        "context_recall": "retrieval_recall",
        "faithfulness": "groundedness",
        "answer_relevancy": "relevancy",
        "factual_correctness": "correctness",
        "answer_correctness": "correctness",
    },
    "custom": {
        "HitRate@1": "ranking",
        "HitRate@K": "ranking",
        "Recall@K": "ranking",
        "Precision@K": "ranking",
        "MRR": "ranking",
        "NDCG@K": "ranking",
        "query_robustness": "robustness",
        "metadata_filtering": "metadata_filtering",
        # Measures routing/HITL-REJECT threshold calibration, not ordinary
        # retrieval quality — deliberately its own dimension, not folded
        # into "ranking" or left unmapped.
        "distance_stratification": "decision_thresholding",
        "llm_faithfulness": "groundedness",
        "llm_correctness": "correctness",
        # The relevancy suite's two distinct checks map to two different
        # dimensions, recorded as two separate metric values for the same
        # suite/case rather than one ambiguous row:
        "required_facts": "completeness",  # deterministic multi-intent coverage check
        "judge_verdict": "relevancy",  # holistic judge verdict
        "llm_hallucination": "hallucination",
        "llm_negative_constraint": "refusal",
        "llm_brand_consistency": "persona",
        "expected_confidence": "confidence_calibration",
    },
    "deepeval": {
        "contextual_relevancy": "retrieval_relevance",
        "contextual_precision": "retrieval_precision",
        "contextual_recall": "retrieval_recall",
        "faithfulness": "groundedness",
        "hallucination": "hallucination",
        "answer_relevancy": "relevancy",
        "geval_factual_correctness": "correctness",
        "geval_answer_correctness": "correctness",
        "geval_negative_constraint": "refusal",
        "geval_brand_consistency": "persona",
    },
}


def get_quality_dimension(framework: str, metric: str) -> "str | None":
    """Returns the normalized quality_dimension for a (framework, metric)
    pair, or None if no honest mapping has been defined for it."""
    return QUALITY_DIMENSIONS.get(framework, {}).get(metric)
