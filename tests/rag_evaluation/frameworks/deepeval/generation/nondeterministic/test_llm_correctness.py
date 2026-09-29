import pytest
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams

from tests.rag_evaluation.frameworks.helpers import get_all_generation_test_cases_from_file, get_or_generate_response
from tests.rag_evaluation.frameworks.deepeval.generation.conftest import (
    record_generation_result,
    resolve_min_score,
)

pytestmark = pytest.mark.live

# Deliberately concise and reusable across every applicable case -- no
# per-case prompt engineering. INPUT/ACTUAL_OUTPUT/EXPECTED_OUTPUT only: this
# criterion checks factual consistency against reference_answer, not
# grounding against retrieval context (that is Faithfulness's job), so
# CONTEXT/RETRIEVAL_CONTEXT is deliberately not in evaluation_params.
FACTUAL_CORRECTNESS_CRITERIA = (
    "Determine whether the 'actual output' is factually consistent with the "
    "'expected output'. Check factual statements, numbers, times, quantities, "
    "and any stated policy conditions or exclusions for direct contradictions "
    "or unsupported alterations. Do not require verbatim wording, and do not "
    "penalize the omission of optional details that are not necessary to "
    "answer the question."
)

# Distinct from FACTUAL_CORRECTNESS_CRITERIA: this checks whether the answer
# as a whole correctly and sufficiently resolves the question (conclusion,
# coverage, completeness), not just whether individual facts are
# contradiction-free. Explicitly scoped away from tone/persona, which
# Custom's brand-consistency suite already covers.
ANSWER_CORRECTNESS_CRITERIA = (
    "Determine whether the 'actual output' correctly and sufficiently answers "
    "the question, using the 'expected output' as the standard of what a "
    "correct answer covers. Check that the overall conclusion is correct, all "
    "essential information is covered, any policy conditions are applied "
    "correctly, and every part of a multi-part question is addressed. "
    "Penalize unsupported conclusions and materially missing information, but "
    "do not penalize tone, phrasing, or persona."
)


@pytest.mark.parametrize(
    "case", get_all_generation_test_cases_from_file("llm_correctness.json"), ids=lambda c: c["case_id"]
)
def test_geval_factual_correctness(
    assistant_llm_service, deepeval_judge_model, assistant_response_cache, recorder, case
):
    """
    GEval Factual Correctness: does the Assistant's real response contain
    any factual contradiction or unsupported alteration relative to
    reference_answer? DeepEval-native counterpart to RAGAS's
    FactualCorrectness -- same correctness quality dimension, different
    algorithm (single holistic Judge rating against an explicit criterion,
    via GEval's evaluation-steps + rubric-scored generation, rather than
    RAGAS's bidirectional claim-decomposition and NLI overlap). Reads
    llm_correctness.json only.
    """
    # Assistant response preparation
    assistant_result = get_or_generate_response(assistant_response_cache, assistant_llm_service, case)
    response = assistant_result["answer"]

    # DeepEval evaluation
    llm_test_case = LLMTestCase(
        input=case["query"],
        actual_output=response,
        expected_output=case["reference_answer"],
    )
    metric = GEval(
        name="Factual Correctness",
        criteria=FACTUAL_CORRECTNESS_CRITERIA,
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.EXPECTED_OUTPUT],
        model=deepeval_judge_model,
        async_mode=False,
    )
    metric.measure(llm_test_case)

    # Reporting
    record_generation_result(
        recorder=recorder,
        case=case,
        metric_name="geval_factual_correctness",
        score=metric.score,
        reason=metric.reason,
        assistant_result=assistant_result,
    )

    # Test validation
    print(f"\nGEvalFactualCorrectness[{case['case_id']}] = {metric.score}")

    assert isinstance(metric.score, (int, float)), (
        f"'{case['case_id']}': expected a numeric GEval Factual Correctness score, got {metric.score!r}"
    )
    assert 0.0 <= metric.score <= 1.0, (
        f"'{case['case_id']}': GEval Factual Correctness score {metric.score} out of [0, 1] range"
    )

    min_score = resolve_min_score(case["evaluation"]["deepeval"], "geval_factual_correctness")
    if min_score is not None:
        assert metric.score >= min_score, (
            f"'{case['case_id']}': GEval Factual Correctness {metric.score:.3f} below minimum {min_score:.3f}"
        )


@pytest.mark.parametrize(
    "case", get_all_generation_test_cases_from_file("llm_correctness.json"), ids=lambda c: c["case_id"]
)
def test_geval_answer_correctness(
    assistant_llm_service, deepeval_judge_model, assistant_response_cache, recorder, case
):
    """
    GEval Answer Correctness: does the Assistant's real response reach the
    right overall conclusion and cover what's essential, relative to
    reference_answer? DeepEval-native counterpart to RAGAS's
    AnswerCorrectness -- same correctness quality dimension, different
    algorithm (single holistic Judge rating against an explicit criterion
    covering conclusion/coverage/completeness, rather than RAGAS's weighted
    combination of statement-level factuality and embedding similarity).
    Reads llm_correctness.json only.
    """
    # Assistant response preparation
    assistant_result = get_or_generate_response(assistant_response_cache, assistant_llm_service, case)
    response = assistant_result["answer"]

    # DeepEval evaluation
    llm_test_case = LLMTestCase(
        input=case["query"],
        actual_output=response,
        expected_output=case["reference_answer"],
    )
    metric = GEval(
        name="Answer Correctness",
        criteria=ANSWER_CORRECTNESS_CRITERIA,
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.EXPECTED_OUTPUT],
        model=deepeval_judge_model,
        async_mode=False,
    )
    metric.measure(llm_test_case)

    # Reporting
    record_generation_result(
        recorder=recorder,
        case=case,
        metric_name="geval_answer_correctness",
        score=metric.score,
        reason=metric.reason,
        assistant_result=assistant_result,
    )

    # Test validation
    print(f"\nGEvalAnswerCorrectness[{case['case_id']}] = {metric.score}")

    assert isinstance(metric.score, (int, float)), (
        f"'{case['case_id']}': expected a numeric GEval Answer Correctness score, got {metric.score!r}"
    )
    assert 0.0 <= metric.score <= 1.0, (
        f"'{case['case_id']}': GEval Answer Correctness score {metric.score} out of [0, 1] range"
    )

    min_score = resolve_min_score(case["evaluation"]["deepeval"], "geval_answer_correctness")
    if min_score is not None:
        assert metric.score >= min_score, (
            f"'{case['case_id']}': GEval Answer Correctness {metric.score:.3f} below minimum {min_score:.3f}"
        )
