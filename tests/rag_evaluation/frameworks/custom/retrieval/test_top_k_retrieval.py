import pytest
from tests.rag_evaluation.frameworks.helpers import get_all_retrieval_test_cases_from_file
from tests.rag_evaluation.frameworks.custom.retrieval.conftest import record_custom_retrieval_result

SUITE_FILE = "top_k_retrieval.json"


@pytest.mark.parametrize("test_case", get_all_retrieval_test_cases_from_file(SUITE_FILE), indirect=True, ids=lambda c: c["name"]
)
def test_top_k_retrieval(vector_db_service, recorder, test_case):
    """
    Concept: HitRate@K — does at least one relevant document show up
    anywhere in the top-K results, not just at rank 1?

    Validates that the case's relevant document (`expected_id`) is present
    within the top-K retrieved results, plus a distance ceiling on the
    top-1 hit. This is HitRate@K, not Recall@K: every case here has exactly
    one relevant document, and with only one relevant document Recall@K
    collapses to the same value as HitRate@K — see
    test_ranking_metrics.py for a genuine multi-relevant
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

    hit = expected_id in retrieved_ids
    record_custom_retrieval_result(
        recorder=recorder, file_name=SUITE_FILE, case_name=test_case["name"], metric_name="HitRate@K",
        score=1.0 if hit else 0.0, threshold=None, passed=hit, query=test_case["query"],
        retrieved_contexts=results.get("documents", [None])[0], expected_ids=[expected_id],
        retrieved_ids=retrieved_ids,
    )
    assert hit, \
       f"Top-k search failure! Expected ID '{expected_id}' not found in top-{top_k}"

    max_distance = test_case["evaluation"]["custom"]["max_distance"]
    distance_ok = distances[0] <= max_distance
    record_custom_retrieval_result(
        recorder=recorder, file_name=SUITE_FILE, case_name=test_case["name"], metric_name="max_distance",
        score=distances[0], threshold=max_distance, passed=distance_ok, query=test_case["query"],
        retrieved_contexts=results.get("documents", [None])[0], expected_ids=[expected_id],
        retrieved_ids=retrieved_ids,
    )
    assert distance_ok, "Top-1 distance is too high!"
