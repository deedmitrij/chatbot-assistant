import pytest
from tests.vector_db.conftest import get_all_test_cases_from_file


@pytest.mark.parametrize("test_case", get_all_test_cases_from_file("test_top_k.json"), indirect=True, ids=lambda x: x[1]["name"]
)
def test_vector_db_top_k(vector_db_service, test_case):
    results = vector_db_service.search(query_text=test_case["query"], n_results=test_case["n_results"])

    retrieved_ids = results['ids'][0]
    distances = results['distances'][0]

    expected_id = test_case['expected_id']
    top_k = test_case["n_results"]

    assert expected_id in retrieved_ids, \
       f"Top-k search failure! Expected ID '{expected_id}' not found in top-{top_k}"

    assert distances[0] <= test_case["max_distance"], "Top-1 distance is too high!"
