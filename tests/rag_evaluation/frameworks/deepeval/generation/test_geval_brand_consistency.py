import pytest
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams

from tests.rag_evaluation.frameworks.deepeval.generation.conftest import (
    get_generation_deepeval_cases,
    get_or_generate_response,
    record_generation_result,
)

pytestmark = pytest.mark.live

# One shared, reusable criterion -- informed by, not a concatenation of, the
# per-case `criteria` strings in llm_brand_consistency.json (first-person
# phrasing, no "as an AI" disclaimer, professionalism under pressure,
# hospitality and warmth). Same pattern as the other GEval metrics: a
# single stable instruction, not per-case prompt engineering.
BRAND_CONSISTENCY_CRITERIA = (
    "Determine whether the 'actual output' consistently represents the "
    "hotel's own official assistant identity and voice. A correct response "
    "speaks in first person as the hotel (e.g. 'we', 'our'), never claims to "
    "be a generic AI or language model, remains polite and professional even "
    "under pressure, and communicates with genuine hospitality and warmth. "
    "Do not evaluate the factual content of the response beyond what is "
    "needed to confirm it stays in character and on-brand."
)


@pytest.mark.parametrize(
    "case", get_generation_deepeval_cases("geval_brand_consistency"), ids=lambda c: c["case_id"]
)
def test_geval_brand_consistency(
    assistant_llm_service, deepeval_judge_model, assistant_response_cache, recorder, case
):
    """
    GEval Brand Consistency: does the Assistant speak in the hotel's own
    voice -- first-person identity, no AI disclaimers, professional tone,
    genuine hospitality? DeepEval-native counterpart to Custom's
    llm_brand_consistency judge-verdict check -- same persona quality
    dimension, different mechanism.

    evaluation_params is INPUT/ACTUAL_OUTPUT only -- no EXPECTED_OUTPUT
    (none of these 4 cases carry a reference_answer, and GEval would raise
    MissingTestCaseParamsError if it were required) and no CONTEXT (this
    criterion evaluates tone/identity, not factual grounding, so pulling in
    context risks the Judge conflating persona with factual correctness,
    which the criterion explicitly excludes).

    Applicability is intentionally the 4 llm_brand_consistency.json cases
    only -- not extended to factual-QA cases elsewhere, even ones with warm
    phrasing, because their primary tested dimension is factual content, not
    hotel identity/voice. The existing case names ("Identity: ...", "Persona:
    ...", "Tone: ...") already distinguish the persona/tone subcategories and
    are preserved as-is through case_name in the reporting record -- no
    reporting-schema change was needed to keep that distinction visible.
    """
    # Assistant response preparation
    assistant_result = get_or_generate_response(assistant_response_cache, assistant_llm_service, case)
    response = assistant_result["answer"]

    # DeepEval evaluation
    llm_test_case = LLMTestCase(
        input=case["query"],
        actual_output=response,
    )
    metric = GEval(
        name="Brand Consistency",
        criteria=BRAND_CONSISTENCY_CRITERIA,
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
        model=deepeval_judge_model,
        async_mode=False,
    )
    metric.measure(llm_test_case)

    # Reporting
    record_generation_result(
        recorder=recorder,
        case=case,
        metric_name="geval_brand_consistency",
        score=metric.score,
        reason=metric.reason,
        assistant_result=assistant_result,
    )

    # Test validation
    print(f"\nGEvalBrandConsistency[{case['case_id']}] = {metric.score}")

    assert isinstance(metric.score, (int, float)), (
        f"'{case['case_id']}': expected a numeric GEval Brand Consistency score, got {metric.score!r}"
    )
    assert 0.0 <= metric.score <= 1.0, (
        f"'{case['case_id']}': GEval Brand Consistency score {metric.score} out of [0, 1] range"
    )

    min_score = case["evaluation"]["deepeval"]["geval_brand_consistency"].get("min_score")
    if min_score is not None:
        assert metric.score >= min_score, (
            f"'{case['case_id']}': GEval Brand Consistency {metric.score:.3f} below minimum {min_score:.3f}"
        )
