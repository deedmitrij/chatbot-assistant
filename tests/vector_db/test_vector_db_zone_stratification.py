import pytest
from tests.vector_db.conftest import get_all_test_cases_from_file
from config import VECTOR_SIMILARITY_THRESHOLD


@pytest.mark.parametrize("test_case", get_all_test_cases_from_file("test_stratification.json"), indirect=True, ids=lambda x: x[1]["name"]
)
def test_vector_db_zone_stratification(vector_db_service, test_case):
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

    assert distance < VECTOR_SIMILARITY_THRESHOLD, \
        f"Distance {distance:.4f} unexpectedly crossed VECTOR_SIMILARITY_THRESHOLD ({VECTOR_SIMILARITY_THRESHOLD})"

    assert test_case["min_dist"] <= distance <= test_case["max_dist"], \
        f"Wrong distance for '{test_case['expected_action']}' action!"
