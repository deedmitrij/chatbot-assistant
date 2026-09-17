import pytest
from ragas.metrics.collections import ContextRelevance, ContextPrecisionWithReference, ContextRecall

from tests.rag_evaluation.frameworks.ragas.retrieval.conftest import (
    get_retrieval_ragas_cases,
    get_single_reference_ragas_cases,
)

pytestmark = pytest.mark.live


def _retrieved_contexts(vector_db_service, test_case):
    n_results = test_case.get("n_results", 3)
    search_result = vector_db_service.search(
        query_text=test_case["query"],
        n_results=n_results,
        where_filter=test_case.get("filter"),
    )
    return search_result["documents"][0]


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


@pytest.mark.parametrize(
    "test_case", get_single_reference_ragas_cases(), ids=lambda item: item[1]["name"], indirect=True
)
def test_context_precision_with_reference(vector_db_service, ragas_judge_llm, test_case):
    """
    ContextPrecisionWithReference: of what we retrieved, how much actually
    supports producing the reference answer? Discriminates against
    topically-relevant-but-factually-wrong hard negatives, which
    ContextRelevance alone can't catch. Only runs on cases whose
    reference_answer is a single, naturally supported factual target (see
    get_single_reference_ragas_cases) — a composite/multi-fact reference,
    where no single chunk covers it in full, is excluded rather than scored
    near-zero regardless of actual retrieval quality.
    """
    retrieved_contexts = _retrieved_contexts(vector_db_service, test_case)
    reference = test_case["reference_answer"]

    result = ContextPrecisionWithReference(llm=ragas_judge_llm).score(
        user_input=test_case["query"], reference=reference, retrieved_contexts=retrieved_contexts
    )

    print(f"\nContextPrecisionWithReference[{test_case['name']}] = {result.value}")

    assert isinstance(result.value, (int, float)), (
        f"'{test_case['name']}': expected a numeric ContextPrecisionWithReference score, got {result.value!r}"
    )
    assert 0.0 <= result.value <= 1.0, (
        f"'{test_case['name']}': ContextPrecisionWithReference score {result.value} out of [0, 1] range"
    )

    min_score = test_case["evaluation"]["ragas"]["context_precision_with_reference"]["min_score"]
    assert result.value >= min_score, (
        f"'{test_case['name']}': ContextPrecisionWithReference {result.value} regressed below min_score {min_score}"
    )


@pytest.mark.parametrize(
    "test_case", get_single_reference_ragas_cases(), ids=lambda item: item[1]["name"], indirect=True
)
def test_context_recall(vector_db_service, ragas_judge_llm, test_case):
    """
    ContextRecall: of what the reference answer needs, how much did we
    actually retrieve? The coverage half of the ContextPrecisionWithReference
    pair, run on the same single-reference case subset for the same reason
    (see get_single_reference_ragas_cases).
    """
    retrieved_contexts = _retrieved_contexts(vector_db_service, test_case)
    reference = test_case["reference_answer"]

    result = ContextRecall(llm=ragas_judge_llm).score(
        user_input=test_case["query"], retrieved_contexts=retrieved_contexts, reference=reference
    )

    print(f"\nContextRecall[{test_case['name']}] = {result.value}")

    assert isinstance(result.value, (int, float)), (
        f"'{test_case['name']}': expected a numeric ContextRecall score, got {result.value!r}"
    )
    assert 0.0 <= result.value <= 1.0, (
        f"'{test_case['name']}': ContextRecall score {result.value} out of [0, 1] range"
    )

    min_score = test_case["evaluation"]["ragas"]["context_recall"]["min_score"]
    assert result.value >= min_score, (
        f"'{test_case['name']}': ContextRecall {result.value} regressed below min_score {min_score}"
    )
