import pytest
from config import JUDGE_MODEL
from tests.rag_evaluation.frameworks.helpers import get_all_generation_test_cases_from_file, get_or_generate_response
from tests.rag_evaluation.frameworks.custom.generation.nondeterministic.conftest import (
    record_custom_generation_result,
    normalize_judge_score,
)

pytestmark = pytest.mark.live

SUITE_FILE = "llm_negative_constraint.json"


@pytest.mark.parametrize("test_case", get_all_generation_test_cases_from_file(SUITE_FILE), ids=lambda x: x["name"])
def test_llm_negative_constraint(assistant_llm_service, llm_as_a_judge, assistant_response_cache, recorder, test_case):
    """
    Refusal/Ignorance: when the query's topic is not addressed by the given
    context at all — an unrelated hotel question, a request entirely outside
    the hotel domain, an empty context, or a context that only partially
    touches the topic without confirming what's asked — the assistant must
    admit it doesn't know rather than guessing or falling back on general
    knowledge. This is the system's main defense against confidently wrong
    answers on off-topic requests — a correct "I don't know" is a PASS, not a
    failure to be helpful.

    This test validates:
    1. (deterministic, always) The confidence flag matches
       `expected_confidence` — always false here, since by construction none
       of these cases have context that supports an answer.
    2. (semantic, via judge) The answer actually communicates the absence of
       information (rather than deflecting, guessing, or inventing an
       answer) in an appropriately polite way.
    3. (semantic, graded) The Judge's normalized score meets this case's
       Golden Dataset min_score, as an additional final check on top of 2.

    This suite has no `required_facts`/`prohibited_patterns` cases:
    "correctly admitting ignorance" has no single exact literal to check
    for, so it stays entirely a judge responsibility beyond confidence.
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

    judge_query = f"""
    Please evaluate the following interaction:
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
        f"Negative Constraint failed: {test_case['name']}\n"
        f"Reason: {judge_verdict['reason']}\n"
        f"Bot's Answer: {bot_answer} (Conf: {bot_confidence})"
    )

    assert score >= min_score, (
        f"Negative Constraint '{test_case['name']}' scored {score:.2f} (raw {judge_verdict['score']}/4), "
        f"below minimum {min_score:.2f}.\n"
    )
