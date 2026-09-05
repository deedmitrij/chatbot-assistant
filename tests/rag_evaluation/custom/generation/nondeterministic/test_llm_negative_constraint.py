import pytest
from tests.rag_evaluation.custom.generation.nondeterministic.conftest import get_all_test_cases_from_file

pytestmark = pytest.mark.live


@pytest.mark.parametrize("test_case", get_all_test_cases_from_file("llm_negative_constraint.json"), ids=lambda x: x["name"])
def test_llm_negative_constraint(llm_as_a_hotel_assistant, llm_as_a_judge, test_case):
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

    This suite has no `required_facts`/`prohibited_patterns` cases:
    "correctly admitting ignorance" has no single exact literal to check
    for, so it stays entirely a judge responsibility beyond confidence.
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

    assert judge_verdict["passed"] is True, (
        f"Negative Constraint failed: {test_case['name']}\n"
        f"Reason: {judge_verdict['reason']}\n"
        f"Bot's Answer: {bot_answer} (Conf: {bot_confidence})"
    )
