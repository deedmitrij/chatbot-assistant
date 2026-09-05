import re
import pytest
from tests.rag_evaluation.custom.generation.nondeterministic.conftest import get_all_test_cases_from_file

pytestmark = pytest.mark.live


@pytest.mark.parametrize("test_case", get_all_test_cases_from_file("llm_hallucination.json"), ids=lambda x: x["name"])
def test_llm_hallucination(llm_as_a_hotel_assistant, llm_as_a_judge, test_case):
    """
    Hallucination: the answer must NOT fabricate a specific detail (a number,
    price, measurement, or attribute) that is absent from context, when that
    detail is queried about an entity/topic the context DOES otherwise
    describe. Every case here is a "tempting fabrication" setup: context is
    relevant and on-topic, but omits exactly the one attribute being asked
    about, creating a realistic incentive for the model to interpolate or
    guess a plausible-sounding value instead of admitting the gap.

    This test validates:
    1. (deterministic, always) The confidence flag matches
       `expected_confidence` — always false here, since no info exists in
       context to support any specific value.
    2. (deterministic, only when the case defines `prohibited_patterns`)
       None of the case's prohibited fabrication patterns (e.g. an invented
       temperature, price, or count) appear in the answer.
    3. (semantic, via judge) The answer correctly admits it lacks the
       requested detail rather than dodging or vaguely deflecting.
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

    for pattern in test_case.get("prohibited_patterns", []):
        assert not re.search(pattern, bot_answer, re.IGNORECASE), (
            f"'{test_case['name']}' appears to have fabricated a value "
            f"matching prohibited pattern '{pattern}'.\n"
            f"Bot Answered: {bot_answer}"
        )

    judge_query = f"""
    Please evaluate the following interaction for HALLUCINATION / FABRICATION:
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
        f"Hallucination check failed: {test_case['name']}\n"
        f"Reason: {judge_verdict['reason']}\n"
        f"Bot's Answer: {bot_answer} (Conf: {bot_confidence})"
    )
