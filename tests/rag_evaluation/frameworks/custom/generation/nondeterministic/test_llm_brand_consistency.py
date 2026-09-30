import re
import pytest
from config import JUDGE_MODEL
from tests.rag_evaluation.frameworks.helpers import get_all_generation_test_cases_from_file, get_or_generate_response
from tests.rag_evaluation.frameworks.custom.generation.nondeterministic.conftest import (
    record_custom_generation_result,
    normalize_judge_score,
)

pytestmark = pytest.mark.live

SUITE_FILE = "llm_brand_consistency.json"


@pytest.mark.parametrize("test_case", get_all_generation_test_cases_from_file(SUITE_FILE), ids=lambda x: x["name"])
def test_llm_brand_consistency(assistant_llm_service, llm_as_a_judge, assistant_response_cache, recorder, test_case):
    """
    Brand/Persona Consistency: the assistant must speak as the hotel itself
    (first-person plural, "Official Hotel Assistant" identity) and never
    break character with AI disclaimers, regardless of how the question is
    phrased. This suite covers persona/tone under normal, non-adversarial
    questions only; adversarial attempts to break the persona (e.g. pressure
    tactics, role-override attempts) belong to a future safety/prompt-
    injection suite and are intentionally excluded here.

    This test validates:
    1. (deterministic, always) The confidence flag matches
       `expected_confidence`.
    2. (deterministic, only when a case defines `required_facts` and/or
       `prohibited_patterns`) A literal fact central to that specific case
       (e.g. "10%") appears, or a persona-breaking phrase (e.g. "as an AI")
       is absent. Most persona cases have neither, because tone/voice isn't
       reducible to a literal check — that's left to the judge.
    3. (semantic, via judge) The tone and phrasing genuinely read as an
       in-character hotel voice rather than a generic assistant.
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

    prohibited_patterns = test_case.get("prohibited_patterns", [])
    matched_patterns = [p for p in prohibited_patterns if re.search(p, bot_answer, re.IGNORECASE)]
    if matched_patterns:
        record_custom_generation_result(
            recorder=recorder, file_name=SUITE_FILE, test_case=test_case, passed=False,
            assistant_answer=bot_answer, actual_confidence=bot_confidence,
            metadata={"failure_stage": "prohibited_patterns", "matched_patterns": matched_patterns},
        )
    assert not matched_patterns, (
        f"'{test_case['name']}' broke persona with a prohibited phrase "
        f"matching {matched_patterns}.\n"
        f"Bot Answered: {bot_answer}"
    )

    judge_query = f"""
    Evaluate the following interaction for TONE, IDENTITY, and HOSPITALITY:
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
        f"Persona check failed: {test_case['name']}\n"
        f"Reason: {judge_verdict['reason']}\n"
        f"Bot's Answer: {bot_answer} (Conf: {bot_confidence})"
    )

    assert score >= min_score, (
        f"Persona check '{test_case['name']}' scored {score:.2f} (raw {judge_verdict['score']}/4), "
        f"below minimum {min_score:.2f}.\n"
    )
