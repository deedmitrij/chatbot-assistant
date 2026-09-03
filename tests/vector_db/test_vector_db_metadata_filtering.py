import pytest
from tests.vector_db.conftest import get_all_test_cases_from_file


@pytest.mark.parametrize("test_case", get_all_test_cases_from_file("test_metadata.json"), indirect=True, ids=lambda x: x[1]["name"])
def test_vector_db_metadata_filtering(vector_db_service, test_case):
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

    assert doc_id == test_case['expected_id'], f"Filter failed! Wrong document"
    assert distance <= test_case["max_distance"], "Distance too high!"
