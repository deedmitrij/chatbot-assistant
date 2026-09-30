import pytest
from config import JUDGE_MODEL
from tests.rag_evaluation.frameworks.helpers import get_all_generation_test_cases_from_file, get_or_generate_response
from tests.rag_evaluation.frameworks.custom.generation.nondeterministic.conftest import (
    record_custom_generation_result,
    normalize_judge_score,
)

pytestmark = pytest.mark.live

SUITE_FILE = "llm_correctness.json"


@pytest.mark.parametrize("test_case", get_all_generation_test_cases_from_file(SUITE_FILE), ids=lambda x: x["name"])
def test_llm_correctness(assistant_llm_service, llm_as_a_judge, assistant_response_cache, recorder, test_case):
    """
    Correctness: the answer must be the right *derived* conclusion, not just
    a restatement of a context fact. This covers applying a stated policy
    rule (e.g. "not free, but $30" from a late-check-out fee rule), comparing
    stated values to recommend the better option, or excluding a case that
    isn't on an allowed list. This is distinct from Faithfulness, which only
    requires the answer to match text already in context verbatim.

    This test validates:
    1. (deterministic, always) The confidence flag matches
       `expected_confidence` — the objectively correct value for a
       rule-derivable answer.
    2. (deterministic, only when the case defines `required_facts`) Any
       literal fact(s) the correct conclusion must cite (e.g. "$30", "$15",
       "$40") appear in the answer.
    3. (semantic, via judge) The reasoning itself is correct — the right
       conclusion was actually reached, not just the right numbers quoted.
    4. (semantic, graded) The Judge's normalized score meets this case's
       Golden Dataset min_score, as an additional final check on top of 3.
    """
    assistant_response = get_or_generate_response(assistant_response_cache, assistant_llm_service, test_case)

    bot_answer = assistant_response["answer"]
    bot_confidence = assistant_response["confidence"]

    confidence_matches = bot_confidence == test_case["expected_confidence"]
    if not confidence_matches:
        record_custom_generation_result(
            recorder=recorder, file_name=SUITE_FILE, test_case=test_case, passed=False,
            assistant_answer=bot_answer, actual_confidence=bot_confidence,
            metadata={"failure_stage": "expected_confidence"},
        )
    assert confidence_matches, (
        f"Confidence calibration failed for '{test_case['name']}': "
        f"expected {test_case['expected_confidence']}, got {bot_confidence}. "
        f"Answer: {bot_answer}"
    )

    required_facts = test_case.get("required_facts", [])
    missing_facts = [fact for fact in required_facts if fact.lower() not in bot_answer.lower()]
    if missing_facts:
        record_custom_generation_result(
            recorder=recorder, file_name=SUITE_FILE, test_case=test_case, passed=False,
            assistant_answer=bot_answer, actual_confidence=bot_confidence,
            metadata={"failure_stage": "required_facts", "missing_facts": missing_facts},
        )
    assert not missing_facts, (
        f"'{test_case['name']}' is missing required fact(s) {missing_facts}.\n"
        f"Bot Answered: {bot_answer}"
    )

    judge_query = f"""
    Please evaluate the following interaction for CORRECTNESS OF REASONING:
    - USER QUERY: {test_case['query']}
    - ASSISTANT ANSWER: {bot_answer}
    - ASSISTANT CONFIDENCE FLAG: {bot_confidence}

    SPECIFIC EVALUATION CRITERIA: {test_case['criteria']}
    """

    judge_verdict = llm_as_a_judge.get_answer(
        query=judge_query,
        context=test_case["context"]
    )

    score = normalize_judge_score(judge_verdict["score"])
    min_score = test_case["evaluation"]["custom"]["min_score"]

    record_custom_generation_result(
        recorder=recorder, file_name=SUITE_FILE, test_case=test_case, passed=judge_verdict["passed"] is True,
        assistant_answer=bot_answer, actual_confidence=bot_confidence,
        judge_reason=judge_verdict["reason"], judge_model=JUDGE_MODEL,
        score=score, threshold=min_score,
    )
    assert judge_verdict["passed"] is True, (
        f"Correctness check failed: {test_case['name']}\n"
        f"Reason: {judge_verdict['reason']}\n"
        f"Bot's Answer: {bot_answer} (Conf: {bot_confidence})"
    )

    assert score >= min_score, (
        f"Correctness check '{test_case['name']}' scored {score:.2f} (raw {judge_verdict['score']}/4), "
        f"below minimum {min_score:.2f}.\n"
    )
