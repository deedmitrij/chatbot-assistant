import pytest
from tests.rag_evaluation.frameworks.helpers import get_all_retrieval_test_cases_from_file
from tests.rag_evaluation.frameworks.custom.retrieval.conftest import record_custom_retrieval_result

SUITE_FILE = "metadata_filtering.json"


@pytest.mark.parametrize("test_case", get_all_retrieval_test_cases_from_file(SUITE_FILE), indirect=True, ids=lambda c: c["name"])
def test_metadata_filtering(vector_db_service, recorder, test_case):
    """
    Concept: metadata filtering — a boundary/access-control check, not a
    ranking metric.

    Validates that `where_filter` correctly restricts which documents are
    even considered before similarity ranking happens — e.g. two documents
    that are semantically near-identical ("guest breakfast" vs "staff
    breakfast") must resolve to different results depending on the `role`
    filter. Intentionally separate from HitRate@K/Recall@K/MRR, which all
    assume the candidate pool is already correct.

    Exists to catch a filter bug that similarity search alone can't reveal:
    the wrong-but-plausible document ranking best only because the filter
    failed to exclude it.
    """
    results = vector_db_service.search(query_text=test_case["query"], where_filter=test_case["filter"])

    doc_id = results['ids'][0][0]
    distance = results['distances'][0][0]

    filter_ok = doc_id == test_case['expected_id']
    record_custom_retrieval_result(
        recorder=recorder, file_name=SUITE_FILE, case_name=test_case["name"], metric_name="metadata_filtering",
        score=1.0 if filter_ok else 0.0, threshold=None, passed=filter_ok, query=test_case["query"],
        retrieved_contexts=results.get("documents", [None])[0], expected_ids=[test_case['expected_id']],
        retrieved_ids=results['ids'][0], filter=test_case["filter"],
    )
    assert filter_ok, "Filter failed! Wrong document"

    max_distance = test_case["evaluation"]["custom"]["max_distance"]
    distance_ok = distance <= max_distance
    record_custom_retrieval_result(
        recorder=recorder, file_name=SUITE_FILE, case_name=test_case["name"], metric_name="max_distance",
        score=distance, threshold=max_distance, passed=distance_ok, query=test_case["query"],
        retrieved_contexts=results.get("documents", [None])[0], expected_ids=[test_case['expected_id']],
        retrieved_ids=results['ids'][0], filter=test_case["filter"],
    )
    assert distance_ok, "Distance too high!"
