import json

from tests.rag_evaluation.reporting.recorder import Recorder
from tests.rag_evaluation.reporting.schema import Status


def _minimal_fields(**overrides):
    fields = dict(
        framework="ragas",
        phase="generation",
        suite="llm_faithfulness.json",
        case_id="llm_faithfulness.json::Pool hours factual check",
        case_name="Pool hours factual check",
        metric="faithfulness",
        status=Status.BASELINE,
        query="What are the pool opening hours?",
        score=1.0,
    )
    fields.update(overrides)
    return fields


def test_record_appends_one_flushed_line(tmp_path):
    events_path = tmp_path / "events.jsonl"
    recorder = Recorder(events_path)

    recorder.record(**_minimal_fields())

    lines = events_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["metric"] == "faithfulness"
    assert parsed["score"] == 1.0


def test_record_auto_populates_identity_fields(tmp_path):
    recorder = Recorder(tmp_path / "events.jsonl")

    result = recorder.record(**_minimal_fields())

    assert result.evaluation_id
    assert result.run_id == recorder.run_id
    assert result.completed_at


def test_successive_records_append_rather_than_overwrite(tmp_path):
    events_path = tmp_path / "events.jsonl"
    recorder = Recorder(events_path)

    recorder.record(**_minimal_fields(case_name="Case A", case_id="a"))
    recorder.record(**_minimal_fields(case_name="Case B", case_id="b"))

    lines = events_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["case_id"] == "a"
    assert json.loads(lines[1])["case_id"] == "b"


def test_each_record_gets_a_distinct_evaluation_id_same_run_id(tmp_path):
    recorder = Recorder(tmp_path / "events.jsonl")

    first = recorder.record(**_minimal_fields(case_id="a"))
    second = recorder.record(**_minimal_fields(case_id="b"))

    assert first.evaluation_id != second.evaluation_id
    assert first.run_id == second.run_id


def test_creates_parent_directory_if_missing(tmp_path):
    events_path = tmp_path / "nested" / "dir" / "events.jsonl"
    recorder = Recorder(events_path)

    recorder.record(**_minimal_fields())

    assert events_path.exists()
