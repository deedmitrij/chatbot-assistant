import pytest
from tests.rag_evaluation.frameworks.helpers import get_all_retrieval_test_cases_from_file
from tests.rag_evaluation.frameworks.custom.retrieval.conftest import record_custom_retrieval_result

SUITE_FILE = "query_robustness.json"


@pytest.mark.parametrize("test_case", get_all_retrieval_test_cases_from_file(SUITE_FILE), indirect=True, ids=lambda c: c["name"])
def test_query_robustness(vector_db_service, recorder, test_case):
    """
    Concept: Top-1 accuracy / HitRate@1 — is the single best-ranked result
    the correct document?

    Validates that across differently-worded versions of the same question
    (clean / noisy / typo / paraphrase — the paraphrase case shares no
    keywords with its target document), the top result stays correct and
    its distance stays under a ceiling. Each case is checked individually,
    not averaged into one score — see test_ranking_metrics.py
    for that.

    Exists to catch regressions where paraphrasing or typos push the
    correct answer out of the top spot, since production only uses the
    LLM's answer when the nearest match is confident (see
    VECTOR_SIMILARITY_THRESHOLD in config.py).
    """
    results = vector_db_service.search(query_text=test_case["query"])

    doc_id = results['ids'][0][0]
    distance = results['distances'][0][0]

    top1_ok = doc_id == test_case['expected_id']
    record_custom_retrieval_result(
        recorder=recorder, file_name=SUITE_FILE, case_name=test_case["name"], metric_name="query_robustness",
        score=1.0 if top1_ok else 0.0, threshold=None, passed=top1_ok, query=test_case["query"],
        retrieved_contexts=results.get("documents", [None])[0], expected_ids=[test_case['expected_id']],
        retrieved_ids=results['ids'][0],
    )
    assert top1_ok, "Wrong document!"

    max_distance = test_case["evaluation"]["custom"]["max_distance"]
    distance_ok = distance <= max_distance
    record_custom_retrieval_result(
        recorder=recorder, file_name=SUITE_FILE, case_name=test_case["name"], metric_name="max_distance",
        score=distance, threshold=max_distance, passed=distance_ok, query=test_case["query"],
        retrieved_contexts=results.get("documents", [None])[0], expected_ids=[test_case['expected_id']],
        retrieved_ids=results['ids'][0],
    )
    assert distance_ok, "Distance too high!"
