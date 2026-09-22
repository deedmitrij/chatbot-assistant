import pytest
from ragas.metrics.collections import ContextRecall

from tests.rag_evaluation.frameworks.ragas.retrieval.conftest import (
    get_single_reference_ragas_cases,
    _retrieved_contexts,
)

pytestmark = pytest.mark.live


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
