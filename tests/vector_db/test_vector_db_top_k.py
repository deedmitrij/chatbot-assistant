import pytest
from tests.vector_db.conftest import get_all_test_cases_from_file


@pytest.mark.parametrize("test_case", get_all_test_cases_from_file("test_top_k.json"), indirect=True, ids=lambda x: x[1]["name"]
)
def test_vector_db_top_k(vector_db_service, test_case):
    """
    Concept: HitRate@K — does at least one relevant document show up
    anywhere in the top-K results, not just at rank 1?

    Validates that the case's relevant document (`expected_id`) is present
    within the top-K retrieved results, plus a distance ceiling on the
    top-1 hit. This is HitRate@K, not Recall@K: every case here has exactly
    one relevant document, and with only one relevant document Recall@K
    collapses to the same value as HitRate@K — see
    test_vector_db_retrieval_metrics.py for a genuine multi-relevant
    Recall@K case and the aggregate metric computation.

    Exists because a correct answer at rank 2 or 3 is still useful context
    for the LLM (KnowledgeManager retrieves top-3 by default) even when
    it's not the single best match — this test is more forgiving than the
    top-1 test above by design.
    """
    results = vector_db_service.search(query_text=test_case["query"], n_results=test_case["n_results"])

    retrieved_ids = results['ids'][0]
    distances = results['distances'][0]

    expected_id = test_case['expected_id']
    top_k = test_case["n_results"]

    assert expected_id in retrieved_ids, \
       f"Top-k search failure! Expected ID '{expected_id}' not found in top-{top_k}"

    assert distances[0] <= test_case["max_distance"], "Top-1 distance is too high!"
