import os
import json
import pytest
from backend.constants import LLMRole
from backend.services.llm.llm_service import LLMService


@pytest.fixture
def llm_as_a_hotel_assistant():
    """Default LLM service that acts as a Hotel Assistant."""
    return LLMService()

@pytest.fixture
def llm_as_a_judge():
    """LLM service that acts as a judge."""
    return LLMService(role=LLMRole.JUDGE)

def get_all_test_cases_from_file(file_name):
    """Utility to load test cases from JSON."""
    file_path = os.path.join(os.path.dirname(__file__), "data", file_name)
    with open(file_path, "r") as f:
        suites = json.load(f)
    test_cases = []
    for suite in suites:
        context = suite["context"]
        for case in suite["cases"]:
            case["context"] = context
            test_cases.append(case)
    return test_cases
