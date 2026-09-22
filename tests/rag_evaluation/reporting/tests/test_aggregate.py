import json

from tests.rag_evaluation.reporting.aggregate import aggregate
from tests.rag_evaluation.reporting.schema import Status

KEY = dict(framework="ragas", phase="generation", case_id="case-1", metric="faithfulness")


def _record(status, completed_at, score=None, **overrides):
    record = dict(KEY)
    record.update(
        evaluation_id=f"eval-{completed_at}",
        run_id="run-1",
        completed_at=completed_at,
        status=status,
        score=score,
        case_name="Case 1",
        suite="llm_faithfulness.json",
        query="q",
    )
    record.update(overrides)
    return record


def _write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def _find_entry(aggregated, **key):
    for entry in aggregated["results"]:
        if entry["key"] == key:
            return entry
    raise AssertionError(f"No entry found for key {key}")


def test_valid_then_newer_valid_keeps_newest_as_both_pointers(tmp_path):
    events_path = tmp_path / "events.jsonl"
    _write_jsonl(
        events_path,
        [
            _record(Status.PASS, "2026-09-17T10:00:00+00:00", score=0.5),
            _record(Status.FAIL, "2026-09-17T11:00:00+00:00", score=0.3),
        ],
    )

    aggregated = aggregate(events_path)
    entry = _find_entry(aggregated, **KEY)

    assert entry["latest_attempt"]["score"] == 0.3
    assert entry["latest_valid"]["score"] == 0.3


def test_valid_then_newer_error_does_not_erase_last_valid_score(tmp_path):
    events_path = tmp_path / "events.jsonl"
    _write_jsonl(
        events_path,
        [
            _record(Status.PASS, "2026-09-17T10:00:00+00:00", score=0.9),
            _record(Status.ERROR, "2026-09-17T11:00:00+00:00", score=None, error_message="Connection error"),
        ],
    )

    aggregated = aggregate(events_path)
    entry = _find_entry(aggregated, **KEY)

    assert entry["latest_attempt"]["status"] == Status.ERROR
    assert entry["latest_attempt"]["error_message"] == "Connection error"
    assert entry["latest_valid"]["status"] == Status.PASS
    assert entry["latest_valid"]["score"] == 0.9


def test_error_then_newer_valid_promotes_valid_to_both_pointers(tmp_path):
    events_path = tmp_path / "events.jsonl"
    _write_jsonl(
        events_path,
        [
            _record(Status.ERROR, "2026-09-17T10:00:00+00:00", score=None),
            _record(Status.PASS, "2026-09-17T11:00:00+00:00", score=0.8),
        ],
    )

    aggregated = aggregate(events_path)
    entry = _find_entry(aggregated, **KEY)

    assert entry["latest_attempt"]["status"] == Status.PASS
    assert entry["latest_attempt"]["score"] == 0.8
    assert entry["latest_valid"]["status"] == Status.PASS
    assert entry["latest_valid"]["score"] == 0.8


def test_na_only_history_leaves_latest_valid_null(tmp_path):
    events_path = tmp_path / "events.jsonl"
    _write_jsonl(
        events_path,
        [
            _record(Status.NOT_APPLICABLE, "2026-09-17T10:00:00+00:00", score=None),
            _record(Status.NOT_APPLICABLE, "2026-09-17T11:00:00+00:00", score=None),
        ],
    )

    aggregated = aggregate(events_path)
    entry = _find_entry(aggregated, **KEY)

    assert entry["latest_attempt"]["completed_at"] == "2026-09-17T11:00:00+00:00"
    assert entry["latest_valid"] is None


def test_malformed_final_line_does_not_break_aggregation(tmp_path):
    events_path = tmp_path / "events.jsonl"
    good_records = [
        _record(Status.PASS, "2026-09-17T10:00:00+00:00", score=0.7),
        _record(Status.PASS, "2026-09-17T11:00:00+00:00", score=0.8),
    ]
    with open(events_path, "w", encoding="utf-8") as f:
        for record in good_records:
            f.write(json.dumps(record) + "\n")
        # Simulate a crash mid-write: a truncated, invalid final line.
        f.write('{"framework": "ragas", "phase": "generation", "case_id": "case-1", "metric": "fait')

    aggregated = aggregate(events_path)
    entry = _find_entry(aggregated, **KEY)

    assert entry["latest_attempt"]["score"] == 0.8
    assert entry["latest_valid"]["score"] == 0.8
    # The truncated line must not produce a spurious extra entry.
    assert len(aggregated["results"]) == 1
