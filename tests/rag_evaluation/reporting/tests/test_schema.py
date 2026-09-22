import json

import pytest

from tests.rag_evaluation.reporting.schema import EvaluationResult, Status


def _minimal_kwargs(**overrides):
    kwargs = dict(
        evaluation_id="eval-1",
        run_id="run-1",
        completed_at="2026-09-17T12:00:00+00:00",
        framework="ragas",
        phase="generation",
        suite="llm_faithfulness.json",
        case_id="llm_faithfulness.json::Pool hours factual check",
        case_name="Pool hours factual check",
        metric="faithfulness",
        status=Status.BASELINE,
        query="What are the pool opening hours?",
    )
    kwargs.update(overrides)
    return kwargs


def test_minimal_construction_defaults_optional_fields_to_none():
    result = EvaluationResult(**_minimal_kwargs())

    assert result.score is None
    assert result.threshold is None
    assert result.error_message is None
    assert result.assistant_answer is None
    assert result.metadata == {}
    assert result.assistant_calls is None
    assert result.estimated_judge_calls is None
    assert result.estimated_embedding_calls is None


def test_invalid_status_raises():
    with pytest.raises(ValueError):
        EvaluationResult(**_minimal_kwargs(status="NOT_A_REAL_STATUS"))


@pytest.mark.parametrize("status", sorted(Status.ALL))
def test_all_declared_statuses_are_accepted(status):
    result = EvaluationResult(**_minimal_kwargs(status=status))
    assert result.status == status


def test_to_dict_round_trips_all_fields():
    result = EvaluationResult(**_minimal_kwargs(score=0.9, assistant_calls=1))
    data = result.to_dict()

    assert data["score"] == 0.9
    assert data["assistant_calls"] == 1
    assert data["case_id"] == "llm_faithfulness.json::Pool hours factual check"


def test_to_json_line_is_valid_single_line_json():
    result = EvaluationResult(**_minimal_kwargs())
    line = result.to_json_line()

    assert "\n" not in line
    parsed = json.loads(line)
    assert parsed["evaluation_id"] == "eval-1"


def test_estimated_call_counters_are_named_explicitly_not_measured():
    # Guards the naming contract from the approved design: these are
    # estimates derived from known metric implementation behavior, never
    # runtime-measured, so the field names must say "estimated_".
    fields = EvaluationResult.__dataclass_fields__
    assert "estimated_judge_calls" in fields
    assert "estimated_embedding_calls" in fields
    assert "judge_calls" not in fields
    assert "embedding_calls" not in fields
