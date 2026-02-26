import os
import json
import pytest
from backend.services.vector_db_service import VectorDBService


@pytest.fixture(scope="session")
def vector_db_service():
    collection = "test_collection"
    db = VectorDBService(collection=collection)
    yield db
    db.client.delete_collection(collection)

def get_all_test_cases_from_file(file_name):
    """Utility to load test cases from JSON."""
    file_path = os.path.join(os.path.dirname(__file__), "data", file_name)
    with open(file_path, "r") as f:
        suites = json.load(f)
    test_cases = []
    for suite in suites:
        for case in suite["cases"]:
            test_cases.append((suite["dataset"], case))
    return test_cases

@pytest.fixture
def test_case(vector_db_service, request):
    """
    Load the dataset to the vector database.
    Pass the test case to the test function.
    """
    dataset, case = request.param
    vector_db_service.upsert_batch(
        documents=dataset["documents"],
        ids=dataset["ids"],
        metadatas=dataset["metadatas"]
    )
    return case
