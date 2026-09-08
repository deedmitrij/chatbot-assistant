import pytest
from tests.rag_evaluation.custom.generation.nondeterministic.conftest import get_all_test_cases_from_file

pytestmark = pytest.mark.live


@pytest.mark.parametrize("test_case", get_all_test_cases_from_file("llm_faithfulness.json"), ids=lambda x: x["name"])
def test_llm_faithfulness(llm_as_a_hotel_assistant, llm_as_a_judge, test_case):
    """
    Faithfulness/Groundedness: the answer must be a faithful restatement of a
    fact that is explicitly present in the retrieved context, with no
    invention and no derivation required. This is the core anti-hallucination
    property of a RAG system — an assistant that drifts from its source
    material erodes guest trust even when it "sounds" plausible.

    This test validates:
    1. (deterministic, always) The confidence flag matches
       `expected_confidence` — the objectively correct value for whether the
       fact is present in context.
    2. (deterministic, only when the case defines `required_facts`) The
       literal fact(s) central to this specific case appear in the answer.
    3. (semantic, via judge) The answer is grounded, on-topic, and phrased
       appropriately — judged against the case's free-text criteria.
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
        f"Test '{test_case['name']}' failed.\n"
        f"Reason: {judge_verdict['reason']}\n"
        f"Bot Answered: {bot_answer} (Confidence: {bot_confidence})"
    )
