import pytest
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams

from tests.rag_evaluation.frameworks.deepeval.generation.conftest import (
    get_generation_deepeval_cases,
    get_or_generate_response,
    record_generation_result,
)

pytestmark = pytest.mark.live

# One shared, reusable criterion -- not a concatenation of the per-case
# `criteria` strings. Those per-case texts informed the wording below (e.g.
# "must NOT invent a temperature", "must not attempt to fulfill the
# request") but are deliberately generalized into a single stable
# instruction, matching the pattern already used for Factual/Answer
# Correctness rather than building a large per-case Judge prompt.
NEGATIVE_CONSTRAINT_CRITERIA = (
    "Determine whether the 'actual output' correctly complies with any "
    "limits, exclusions, or missing information relevant to the 'input', "
    "given the 'context'. A correct response acknowledges what is "
    "unavailable, unsupported, or outside scope rather than guessing, "
    "inventing an alternative, or converting missing information into an "
    "unsupported positive claim. Do not penalize an appropriately worded "
    "refusal or absence-of-information response."
)


@pytest.mark.parametrize(
    "case", get_generation_deepeval_cases("geval_negative_constraint"), ids=lambda c: c["case_id"]
)
def test_geval_negative_constraint(
    assistant_llm_service, deepeval_judge_model, assistant_response_cache, recorder, case
):
    """
    GEval Negative Constraint: does the Assistant correctly decline,
    exclude, or admit missing information instead of fabricating an
    unsupported answer? DeepEval-native counterpart to Custom's
    llm_negative_constraint judge-verdict check -- same refusal quality
    dimension, different mechanism (GEval's single holistic rating against
    an explicit criterion, rather than a free-form judge_verdict pass/fail).

    CONTEXT is included in evaluation_params (beyond the INPUT/ACTUAL_OUTPUT
    the task prefers by default) because the criterion explicitly turns on
    "invented alternative" / "converting missing information into an
    unsupported claim" -- verifying that requires knowing what information
    actually was or wasn't available, which only `context` can tell the
    Judge. GEval's single holistic prompt handles the empty-context case
    (llm_negative_constraint.json's "Empty Context Test") gracefully -- it
    only fails structurally when required_output is missing, not on an
    empty list -- unlike HallucinationMetric's QAG loop.

    Applicability is intentionally NOT limited to llm_negative_constraint.json's
    4 cases: llm_hallucination.json's 4 "Hallucination Trap" cases are also
    included, because their own `criteria` text ("must NOT invent a
    temperature/price/count/fee") is squarely a "not converting unavailable
    information into an unsupported positive claim" scenario -- one of this
    metric's explicitly named behaviors, not a stretch. Cases from
    llm_correctness.json that also involve a denial (e.g. the pet-exclusion
    case) are deliberately excluded: there, all the information needed to
    answer is already present in context, so the test is about correctly
    *deriving a conclusion* (Correctness), not about behavior under missing
    or restricted information (Negative Constraint).
    """
    # Assistant response preparation
    assistant_result = get_or_generate_response(assistant_response_cache, assistant_llm_service, case)
    response = assistant_result["answer"]

    # DeepEval evaluation
    llm_test_case = LLMTestCase(
        input=case["query"],
        actual_output=response,
        context=case["context"],
    )
    metric = GEval(
        name="Negative Constraint",
        criteria=NEGATIVE_CONSTRAINT_CRITERIA,
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.CONTEXT],
        model=deepeval_judge_model,
        async_mode=False,
    )
    metric.measure(llm_test_case)

    # Reporting
    record_generation_result(
        recorder=recorder,
        case=case,
        metric_name="geval_negative_constraint",
        score=metric.score,
        reason=metric.reason,
        assistant_result=assistant_result,
    )

    # Test validation
    print(f"\nGEvalNegativeConstraint[{case['case_id']}] = {metric.score}")

    assert isinstance(metric.score, (int, float)), (
        f"'{case['case_id']}': expected a numeric GEval Negative Constraint score, got {metric.score!r}"
    )
    assert 0.0 <= metric.score <= 1.0, (
        f"'{case['case_id']}': GEval Negative Constraint score {metric.score} out of [0, 1] range"
    )

    min_score = case["evaluation"]["deepeval"]["geval_negative_constraint"].get("min_score")
    if min_score is not None:
        assert metric.score >= min_score, (
            f"'{case['case_id']}': GEval Negative Constraint {metric.score:.3f} below minimum {min_score:.3f}"
        )
