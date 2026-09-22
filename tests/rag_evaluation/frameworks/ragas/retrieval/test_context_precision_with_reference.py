import pytest
from ragas.metrics.collections import ContextPrecisionWithReference

from tests.rag_evaluation.frameworks.ragas.retrieval.conftest import (
    get_single_reference_ragas_cases,
    _retrieved_contexts,
)

pytestmark = pytest.mark.live


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
