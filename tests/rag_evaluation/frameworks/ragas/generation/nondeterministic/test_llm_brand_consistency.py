import pytest
from ragas.metrics.collections import InstanceSpecificRubrics

from tests.rag_evaluation.frameworks.helpers import get_all_generation_test_cases_from_file, get_or_generate_response
from tests.rag_evaluation.frameworks.ragas.generation.conftest import (
    record_generation_result,
    build_instance_rubrics,
    normalize_rubric_score,
    RubricResult,
)
from tests.rag_evaluation.frameworks.ragas.conftest import resolve_min_score

pytestmark = pytest.mark.live


@pytest.mark.parametrize(
    "case", get_all_generation_test_cases_from_file("llm_brand_consistency.json"), ids=lambda c: c["case_id"]
)
def test_brand_consistency_rubric(assistant_llm_service, ragas_judge_llm, assistant_response_cache, recorder, case):
    """
    InstanceSpecificRubrics: does the Assistant's real response stay in the
    hotel's persona/tone, per this case's own criteria? RAGAS-native
    counterpart to Custom's llm_brand_consistency graded Judge -- same
    persona quality dimension, RAGAS's own per-case rubric mechanism
    instead of a free-form Judge prompt. Reads llm_brand_consistency.json
    only.
    """
    assistant_result = get_or_generate_response(assistant_response_cache, assistant_llm_service, case)
    response = assistant_result["answer"]

    # RAGAS evaluation
    rubrics = build_instance_rubrics(case["criteria"])
    metric = InstanceSpecificRubrics(llm=ragas_judge_llm)
    result = metric.score(
        user_input=case["query"], response=response, retrieved_contexts=case["context"], rubrics=rubrics
    )

    # Validate the native RAGAS result before normalizing it
    assert isinstance(result.value, (int, float)), (
        f"'{case['case_id']}': expected a numeric InstanceSpecificRubrics score, got {result.value!r}"
    )
    assert 1.0 <= result.value <= 5.0, (
        f"'{case['case_id']}': InstanceSpecificRubrics score {result.value} out of [1, 5] range"
    )

    normalized_value = normalize_rubric_score(result.value)

    # Reporting
    record_generation_result(
        recorder=recorder,
        case=case,
        metric_name="brand_consistency",
        result=RubricResult(value=normalized_value, reason=result.reason),
        assistant_result=assistant_result,
        metadata={"raw_rubric_score": result.value},
    )

    min_score = resolve_min_score(case["evaluation"]["ragas"], "brand_consistency")
    if min_score is not None:
        assert normalized_value >= min_score, (
            f"'{case['case_id']}': Brand Consistency rubric score {normalized_value:.3f} "
            f"(raw {result.value}/5) below minimum {min_score:.3f}"
        )
