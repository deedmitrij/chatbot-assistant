import pytest
from tests.vector_db.conftest import get_all_test_cases_from_file


@pytest.mark.parametrize("test_case", get_all_test_cases_from_file("test_retrieval.json"), indirect=True, ids=lambda x: x[1]["name"])
def test_vector_db_retrieval(vector_db_service, test_case):
    results = vector_db_service.search(query_text=test_case["query"])

    doc_id = results['ids'][0][0]
    distance = results['distances'][0][0]

    assert doc_id == test_case['expected_id'], "Wrong document!"
    assert distance <= test_case["max_distance"], "Distance too high!"
