import pytest
from tests.rag_evaluation.frameworks.helpers import get_all_retrieval_test_cases_from_file
from tests.rag_evaluation.frameworks.custom.retrieval.conftest import record_custom_retrieval_result
from config import VECTOR_SIMILARITY_THRESHOLD

SUITE_FILE = "distance_stratification.json"


@pytest.mark.parametrize("test_case", get_all_retrieval_test_cases_from_file(SUITE_FILE), indirect=True, ids=lambda c: c["name"]
)
def test_distance_stratification(vector_db_service, recorder, test_case):
    """
    Concept: distance/threshold calibration — not a ranking metric, a sanity
    check on the raw distance numbers the confidence gate relies on.

    Validates two things about known-ambiguous ("gray zone") and
    known-out-of-scope ("garbage") queries: (1) their nearest-document
    distance falls within an expected band, calibrated from measured
    values, not guessed; and (2) that distance stays below
    VECTOR_SIMILARITY_THRESHOLD (config.py) — meaning, for this dataset,
    distance alone never crosses the production confidence gate on its own.
    This test does NOT invoke ChatManager or assert the `expected_action`
    (HITL/REJECT) label; final routing also depends on LLM confidence,
    which is covered separately in the ChatManager tests.

    Exists to make the retrieval-distance side of the confidence gate
    honest rather than assumed: these queries are measurably farther from
    any document than a confident topical match, but not far enough to be
    rejected by distance alone — so if this dataset's ambiguous/garbage
    inputs get escalated to a human in production, it's the LLM's
    confidence doing that work, not the vector distance.
    """
    results = vector_db_service.search(query_text=test_case["query"], n_results=1)

    distance = results['distances'][0][0]

    below_global_threshold = distance < VECTOR_SIMILARITY_THRESHOLD
    record_custom_retrieval_result(
        recorder=recorder, file_name=SUITE_FILE, case_name=test_case["name"], metric_name="vector_similarity_threshold",
        score=distance, threshold=VECTOR_SIMILARITY_THRESHOLD, passed=below_global_threshold, query=test_case["query"],
        retrieved_contexts=results.get("documents", [None])[0], retrieved_ids=results['ids'][0],
    )
    assert below_global_threshold, \
        f"Distance {distance:.4f} unexpectedly crossed VECTOR_SIMILARITY_THRESHOLD ({VECTOR_SIMILARITY_THRESHOLD})"

    custom_evaluation = test_case["evaluation"]["custom"]
    band_ok = custom_evaluation["min_dist"] <= distance <= custom_evaluation["max_dist"]
    record_custom_retrieval_result(
        recorder=recorder, file_name=SUITE_FILE, case_name=test_case["name"], metric_name="distance_stratification",
        score=distance, threshold=custom_evaluation["max_dist"], passed=band_ok, query=test_case["query"],
        retrieved_contexts=results.get("documents", [None])[0], retrieved_ids=results['ids'][0],
        metadata={
            "min_dist": custom_evaluation["min_dist"],
            "max_dist": custom_evaluation["max_dist"],
            "expected_action": test_case.get("expected_action"),
        },
    )
    assert band_ok, f"Wrong distance for '{test_case['expected_action']}' action!"
