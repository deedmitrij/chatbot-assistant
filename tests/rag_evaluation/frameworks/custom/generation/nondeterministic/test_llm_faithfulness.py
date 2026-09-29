import pytest
from config import JUDGE_MODEL
from tests.rag_evaluation.frameworks.helpers import get_all_generation_test_cases_from_file, get_or_generate_response
from tests.rag_evaluation.frameworks.custom.generation.nondeterministic.conftest import (
    record_custom_generation_result,
)

pytestmark = pytest.mark.live

SUITE_FILE = "llm_faithfulness.json"


@pytest.mark.parametrize("test_case", get_all_generation_test_cases_from_file(SUITE_FILE), ids=lambda x: x["name"])
def test_llm_faithfulness(assistant_llm_service, llm_as_a_judge, assistant_response_cache, recorder, test_case):
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

    record_custom_generation_result(
        recorder=recorder, file_name=SUITE_FILE, test_case=test_case, passed=judge_verdict["passed"] is True,
        assistant_answer=bot_answer, actual_confidence=bot_confidence,
        judge_reason=judge_verdict["reason"], judge_model=JUDGE_MODEL,
    )
    assert judge_verdict["passed"] is True, (
        f"Test '{test_case['name']}' failed.\n"
        f"Reason: {judge_verdict['reason']}\n"
        f"Bot Answered: {bot_answer} (Confidence: {bot_confidence})"
    )
