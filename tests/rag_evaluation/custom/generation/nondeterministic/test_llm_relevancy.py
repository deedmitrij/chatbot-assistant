import pytest
from tests.rag_evaluation.custom.generation.nondeterministic.conftest import get_all_test_cases_from_file

pytestmark = pytest.mark.live


@pytest.mark.parametrize("test_case", get_all_test_cases_from_file("llm_relevancy.json"), ids=lambda x: x["name"])
def test_llm_relevancy_and_completeness(llm_as_a_hotel_assistant, llm_as_a_judge, test_case):
    """
    Answer Relevancy/Completeness: when a user asks a multi-part question,
    the answer must address every part, not just the easiest or first one.
    This matters for a support bot because a partially-answered question
    often reads as a full answer to the user, who then never follows up on
    the dropped part.

    This test validates:
    1. (deterministic, always) The confidence flag matches
       `expected_confidence` — both sub-answers are grounded in context in
       these cases, so it is always true.
    2. (deterministic, only when the case defines `required_facts`) The
       literal fact(s) tied to each sub-question appear in the answer.
    3. (semantic, via judge) The answer is complete and coherent as a
       response to the whole multi-part query, not just a keyword match.
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
    Please evaluate the following interaction for RELEVANCY and COMPLETENESS:
    - USER QUERY: {test_case['query']}
    - ASSISTANT ANSWER: {bot_answer}
    - ASSISTANT CONFIDENCE FLAG: {bot_confidence}

    EVALUATION CRITERIA: {test_case['criteria']}
    """

    judge_verdict = llm_as_a_judge.get_answer(
        query=judge_query,
        context=test_case["context"]
    )

    assert judge_verdict["passed"] is True, (
        f"Relevancy/Completeness failed: {test_case['name']}\n"
        f"Reason: {judge_verdict['reason']}\n"
        f"Bot's Answer: {bot_answer} (Conf: {bot_confidence})"
    )
