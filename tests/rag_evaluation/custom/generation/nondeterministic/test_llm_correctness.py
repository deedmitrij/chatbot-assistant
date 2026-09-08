import pytest
from tests.rag_evaluation.custom.generation.nondeterministic.conftest import get_all_test_cases_from_file

pytestmark = pytest.mark.live


@pytest.mark.parametrize("test_case", get_all_test_cases_from_file("llm_correctness.json"), ids=lambda x: x["name"])
def test_llm_correctness(llm_as_a_hotel_assistant, llm_as_a_judge, test_case):
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

    assert judge_verdict["passed"] is True, (
        f"Correctness check failed: {test_case['name']}\n"
        f"Reason: {judge_verdict['reason']}\n"
        f"Bot's Answer: {bot_answer} (Conf: {bot_confidence})"
    )
