import json
from config import RAG_EVALUATION_DATA_DIR


def get_all_generation_test_cases_from_file(file_name):
    """Loads every case from one Generation Golden Dataset file, flattening
    each suite's shared `context` onto its cases and stamping the stable
    case_id: f"{file_name}::{case['name']}". Loads only the file named --
    never scans or combines datasets."""
    file_path = RAG_EVALUATION_DATA_DIR / "generation" / file_name
    with open(file_path, "r") as f:
        suites = json.load(f)

    test_cases = []
    for suite in suites:
        context = suite["context"]
        for case in suite["cases"]:
            case["context"] = context
            case["case_id"] = f"{file_name}::{case['name']}"
            test_cases.append(case)
    return test_cases


def get_all_retrieval_test_cases_from_file(file_name):
    """Loads every case from one Retrieval Golden Dataset file, flattening
    each suite's shared `dataset` onto its cases and stamping the same
    case_id convention as Generation. Loads only the file named -- metric-
    specific eligibility filtering stays local to the test file that needs
    it, not here."""
    file_path = RAG_EVALUATION_DATA_DIR / "retrieval" / file_name
    with open(file_path, "r") as f:
        suites = json.load(f)

    test_cases = []
    for suite in suites:
        dataset = suite["dataset"]
        for case in suite["cases"]:
            case["dataset"] = dataset
            case["case_id"] = f"{file_name}::{case['name']}"
            test_cases.append(case)
    return test_cases


def get_or_generate_response(assistant_response_cache, llm_service, case):
    """Returns a cached Assistant response for this case, generating and
    caching it once per case_id if not already present."""
    case_id = case["case_id"]
    if case_id not in assistant_response_cache:
        assistant_response_cache[case_id] = llm_service.get_answer(query=case["query"], context=case["context"])
    return assistant_response_cache[case_id]
