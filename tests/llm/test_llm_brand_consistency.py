import pytest
from tests.llm.conftest import get_all_test_cases_from_file


@pytest.mark.parametrize("test_case", get_all_test_cases_from_file("llm_brand_consistency.json"), ids=lambda x: x["name"])
def test_llm_brand_consistency(llm_as_a_hotel_assistant, llm_as_a_judge, test_case):
    """
    Evaluates if the assistant maintains the correct hotel persona and tone.
    """
    assistant_response = llm_as_a_hotel_assistant.get_answer(
        query=test_case["query"],
        context=test_case["context"]
    )
    bot_answer = assistant_response['answer']
    bot_confidence = assistant_response['confidence']

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
