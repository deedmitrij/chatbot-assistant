import pytest
from tests.vector_db.conftest import get_all_test_cases_from_file


@pytest.mark.parametrize("test_case", get_all_test_cases_from_file("test_stratification.json"), indirect=True, ids=lambda x: x[1]["name"]
)
def test_vector_db_zone_stratification(vector_db_service, test_case):
    results = vector_db_service.search(query_text=test_case["query"], n_results=1)

    distance = results['distances'][0][0]

    assert test_case["min_dist"] <= distance <= test_case["max_dist"], \
        f"Wrong distance for '{test_case['expected_action']}' action!"
