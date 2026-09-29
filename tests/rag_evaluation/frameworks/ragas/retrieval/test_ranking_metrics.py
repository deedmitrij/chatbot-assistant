import pytest
from ragas.metrics.collections import ContextPrecisionWithReference, ContextRecall

from tests.rag_evaluation.frameworks.helpers import get_all_retrieval_test_cases_from_file
from tests.rag_evaluation.frameworks.ragas.retrieval.conftest import (
    search_retrieval,
    record_retrieval_result,
)

pytestmark = pytest.mark.live

# ContextPrecisionWithReference/ContextRecall judge each retrieved chunk's
# support for the *whole* reference answer, so composite-reference cases (a
# reference synthesizing facts from more than one expected document) are
# structurally ineligible here: no single chunk can support a multi-fact
# reference in full, so these metrics would score them near-zero regardless
# of actual retrieval quality. Those cases stay fully covered by Custom's
# ranking metrics.
RANKING_CASES = [
    case for case in get_all_retrieval_test_cases_from_file("ranking_metrics.json")
    if not case.get("evaluation", {}).get("reference_answer_is_composite", False)
]


@pytest.mark.parametrize("test_case", RANKING_CASES, ids=lambda c: c["name"], indirect=True)
def test_context_precision_with_reference(vector_db_service, ragas_judge_llm, recorder, test_case):
    """
    ContextPrecisionWithReference: of what we retrieved, how much actually
    supports producing the reference answer? Discriminates against
    topically-relevant-but-factually-wrong hard negatives, which
    ContextRelevance alone can't catch. Reads ranking_metrics.json only,
    excluding its composite-reference cases (see RANKING_CASES above).
    """
    # Retrieval data preparation
    retrieved_contexts, retrieved_ids = search_retrieval(vector_db_service, test_case)
    reference = test_case["reference_answer"]

    # RAGAS evaluation
    result = ContextPrecisionWithReference(llm=ragas_judge_llm).score(
        user_input=test_case["query"], reference=reference, retrieved_contexts=retrieved_contexts
    )

    # Reporting
    record_retrieval_result(
        recorder=recorder,
        test_case=test_case,
        metric_name="context_precision_with_reference",
        result=result,
        retrieved_contexts=retrieved_contexts,
        retrieved_ids=retrieved_ids,
    )

    # Test validation
    print(f"\nContextPrecisionWithReference[{test_case['name']}] = {result.value}")

    assert isinstance(result.value, (int, float)), (
        f"'{test_case['name']}': expected a numeric ContextPrecisionWithReference score, got {result.value!r}"
    )
    assert 0.0 <= result.value <= 1.0, (
        f"'{test_case['name']}': ContextPrecisionWithReference score {result.value} out of [0, 1] range"
    )

    min_score = test_case["evaluation"]["ragas"]["min_score"]["context_precision_with_reference"]
    assert result.value >= min_score, (
        f"'{test_case['name']}': ContextPrecisionWithReference {result.value} regressed below min_score {min_score}"
    )


@pytest.mark.parametrize("test_case", RANKING_CASES, ids=lambda c: c["name"], indirect=True)
def test_context_recall(vector_db_service, ragas_judge_llm, recorder, test_case):
    """
    ContextRecall: of what the reference answer needs, how much did we
    actually retrieve? The coverage half of the ContextPrecisionWithReference
    pair, run on the same ranking_metrics.json case subset for the same
    reason (see RANKING_CASES above).
    """
    # Retrieval data preparation
    retrieved_contexts, retrieved_ids = search_retrieval(vector_db_service, test_case)
    reference = test_case["reference_answer"]

    # RAGAS evaluation
    result = ContextRecall(llm=ragas_judge_llm).score(
        user_input=test_case["query"], retrieved_contexts=retrieved_contexts, reference=reference
    )

    # Reporting
    record_retrieval_result(
        recorder=recorder,
        test_case=test_case,
        metric_name="context_recall",
        result=result,
        retrieved_contexts=retrieved_contexts,
        retrieved_ids=retrieved_ids,
    )

    # Test validation
    print(f"\nContextRecall[{test_case['name']}] = {result.value}")

    assert isinstance(result.value, (int, float)), (
        f"'{test_case['name']}': expected a numeric ContextRecall score, got {result.value!r}"
    )
    assert 0.0 <= result.value <= 1.0, (
        f"'{test_case['name']}': ContextRecall score {result.value} out of [0, 1] range"
    )

    min_score = test_case["evaluation"]["ragas"]["min_score"]["context_recall"]
    assert result.value >= min_score, (
        f"'{test_case['name']}': ContextRecall {result.value} regressed below min_score {min_score}"
    )
