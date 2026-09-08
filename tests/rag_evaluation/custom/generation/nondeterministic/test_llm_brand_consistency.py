import re
import pytest
from tests.rag_evaluation.custom.generation.nondeterministic.conftest import get_all_test_cases_from_file

pytestmark = pytest.mark.live


@pytest.mark.parametrize("test_case", get_all_test_cases_from_file("llm_brand_consistency.json"), ids=lambda x: x["name"])
def test_llm_brand_consistency(llm_as_a_hotel_assistant, llm_as_a_judge, test_case):
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
    """
    assistant_response = llm_as_a_hotel_assistant.get_answer(
        query=test_case["query"],
        context=test_case["context"]
    )

    bot_answer = assistant_response["answer"]
    bot_confidence = assistant_response["confidence"]

    assert bot_confidence == test_case["expected_confidence"], (
        f"Confidence calibration failed for '{test_case['name']}': "
        f"expected {test_case['expected_confidence']}, got {bot_confidence}. "
        f"Answer: {bot_answer}"
    )

    for fact in test_case.get("required_facts", []):
        assert fact.lower() in bot_answer.lower(), (
            f"'{test_case['name']}' is missing required fact '{fact}'.\n"
            f"Bot Answered: {bot_answer}"
        )

    for pattern in test_case.get("prohibited_patterns", []):
        assert not re.search(pattern, bot_answer, re.IGNORECASE), (
            f"'{test_case['name']}' broke persona with a prohibited phrase "
            f"matching '{pattern}'.\n"
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

    assert judge_verdict["passed"] is True, (
        f"Persona check failed: {test_case['name']}\n"
        f"Reason: {judge_verdict['reason']}\n"
        f"Bot's Answer: {bot_answer} (Conf: {bot_confidence})"
    )
