import json

import pytest

from reporting.dashboard.data import (
    ReportLoadError,
    build_dashboard,
    find_disagreements,
    group_by_dimension,
    load_report,
    pass_rate,
    select_current_runs,
    split_valid_entries,
    summarize,
    to_row,
)


def _record(framework="custom", phase="generation", suite="llm_correctness.json", case="Case A", metric="llm_correctness",
            status="PASS", run_id="run-1", completed_at="2026-09-30T09:00:00+00:00", **overrides):
    record = dict(
        evaluation_id=f"{framework}-{metric}-{case}-{run_id}",
        run_id=run_id,
        completed_at=completed_at,
        framework=framework,
        phase=phase,
        suite=suite,
        case_id=f"{suite}::{case}",
        case_name=case,
        metric=metric,
        status=status,
        query="q",
        quality_dimension="correctness",
        score=1.0 if status == "PASS" else 0.0,
        threshold=0.75,
        metadata={},
    )
    record.update(overrides)
    return record


def _entry(record, latest_valid="same"):
    return {
        "key": {k: record[k] for k in ("framework", "phase", "case_id", "metric")},
        "latest_attempt": record,
        "latest_valid": record if latest_valid == "same" else latest_valid,
    }


def _write_latest(path, entries):
    path.write_text(json.dumps({"generated_at": "2026-09-30T10:00:00+00:00", "results": entries}), encoding="utf-8")


# ---------- loading ----------

def test_load_report_missing_files_returns_empty_report(tmp_path):
    report = load_report(tmp_path / "latest.json", tmp_path / "events.jsonl")
    assert report == {"source": None, "generated_at": None, "results": []}


def test_load_report_falls_back_to_in_memory_aggregate_of_events(tmp_path):
    events = tmp_path / "events.jsonl"
    events.write_text(json.dumps(_record()) + "\n", encoding="utf-8")
    latest = tmp_path / "latest.json"

    report = load_report(latest, events)

    assert len(report["results"]) == 1
    assert "events.jsonl" in report["source"]
    assert not latest.exists(), "dashboard must never write report files"


def test_load_report_malformed_latest_json_raises_clear_error(tmp_path):
    latest = tmp_path / "latest.json"
    latest.write_text("{not json", encoding="utf-8")
    with pytest.raises(ReportLoadError, match="latest.json"):
        load_report(latest, tmp_path / "events.jsonl")


def test_load_report_latest_json_without_results_list_raises(tmp_path):
    latest = tmp_path / "latest.json"
    latest.write_text(json.dumps({"generated_at": "x"}), encoding="utf-8")
    with pytest.raises(ReportLoadError, match="results"):
        load_report(latest, tmp_path / "events.jsonl")


# ---------- malformed entries / optional fields ----------

def test_malformed_entries_are_skipped_with_warnings_not_fatal():
    good = _entry(_record())
    missing_status = _entry(_record())
    del missing_status["latest_attempt"]["status"]
    bad_status = _entry(_record(case="B"))
    bad_status["latest_attempt"]["status"] = "MAYBE"

    valid, warnings = split_valid_entries([good, "junk", {"latest_attempt": None}, missing_status, bad_status])

    assert valid == [good]
    assert len(warnings) == 4
    assert any("status" in w for w in warnings)


def test_to_row_tolerates_missing_optional_fields():
    minimal = {k: _record()[k] for k in ("evaluation_id", "run_id", "framework", "phase", "suite", "case_id", "metric", "status")}

    row = to_row({"latest_attempt": minimal, "latest_valid": None})

    assert row["metadata"] == {}
    assert row["native_score"] is None
    assert row["is_aggregate"] is False
    assert row["previous_valid"] is None


def test_to_row_exposes_native_score_and_previous_valid():
    valid = _record(run_id="old", score=0.5, status="FAIL")
    error = _record(run_id="new", status="ERROR", score=None, metadata={"raw_rubric_score": 4.0})

    row = to_row(_entry(error, latest_valid=valid))

    assert row["status"] == "ERROR", "latest attempt status must not be replaced by the last valid one"
    assert row["native_score"] == {"name": "raw_rubric_score", "value": 4.0}
    assert row["previous_valid"]["run_id"] == "old"
    assert row["previous_valid"]["score"] == 0.5


def test_aggregate_ranking_rows_are_flagged():
    aggregate = _record(framework="custom", phase="retrieval", suite="ranking_metrics.json", metric="MRR",
                        case="Aggregate (all ranking-metrics queries)", query="[aggregate over 7 queries]")
    assert to_row(_entry(aggregate))["is_aggregate"] is True


# ---------- current run selection ----------

def test_current_run_selection_drops_superseded_entries_of_same_suite():
    old_retired_metric = _entry(_record(metric="judge_verdict", run_id="old", completed_at="2026-09-26T09:00:00+00:00"))
    old_same_metric = _entry(_record(case="Dropped case", run_id="old", completed_at="2026-09-26T09:00:01+00:00"))
    new = _entry(_record(run_id="new", completed_at="2026-09-30T09:00:00+00:00"))

    selected, current = select_current_runs([old_retired_metric, old_same_metric, new])

    assert selected == [new]
    assert current == {("custom", "generation", "llm_correctness.json"): "new"}


def test_current_run_selection_is_per_framework_phase_suite():
    custom = _entry(_record(framework="custom", run_id="c", completed_at="2026-09-30T09:00:00+00:00"))
    ragas_older = _entry(_record(framework="ragas", metric="answer_correctness", run_id="r", completed_at="2026-09-25T09:00:00+00:00"))
    other_suite = _entry(_record(suite="llm_relevancy.json", metric="llm_relevancy", run_id="old", completed_at="2026-09-20T09:00:00+00:00"))

    selected, _ = select_current_runs([custom, ragas_older, other_suite])

    assert selected == [custom, ragas_older, other_suite]


# ---------- summaries ----------

def test_pass_rate_excludes_ungated_and_handles_zero():
    assert pass_rate(3, 1) == 0.75
    assert pass_rate(0, 0) is None


def test_summarize_counts_by_framework_status_and_phase():
    rows = [to_row(_entry(r)) for r in (
        _record(framework="custom", case="A", status="PASS"),
        _record(framework="custom", case="B", status="FAIL"),
        _record(framework="ragas", metric="context_recall", phase="retrieval", case="A", status="PASS"),
        _record(framework="deepeval", metric="faithfulness", case="A", status="BASELINE", threshold=None),
    )]

    summary = summarize(rows)

    overall = summary["overall"]
    assert (overall["total"], overall["pass"], overall["fail"], overall["other"]) == (4, 2, 1, 1)
    assert overall["pass_rate"] == pytest.approx(2 / 3)
    assert overall["frameworks"] == ["custom", "deepeval", "ragas"]
    assert overall["by_phase"] == {"generation": 3, "retrieval": 1}
    assert summary["frameworks"]["custom"]["pass_rate"] == 0.5
    assert summary["frameworks"]["deepeval"]["pass_rate"] is None
    assert summary["frameworks"]["deepeval"]["by_status"] == {"BASELINE": 1}


def test_group_by_dimension_keeps_frameworks_and_metrics_separate():
    rows = [to_row(_entry(r)) for r in (
        _record(framework="custom", metric="llm_correctness", case="A", status="PASS"),
        _record(framework="ragas", metric="answer_correctness", case="A", status="FAIL"),
        _record(framework="ragas", metric="factual_correctness", case="A", status="PASS"),
        _record(framework="custom", phase="retrieval", metric="max_distance", case="A", quality_dimension=None),
    )]

    groups = {(g["phase"], g["quality_dimension"]): g for g in group_by_dimension(rows)}

    correctness = groups[("generation", "correctness")]["frameworks"]
    assert correctness["custom"]["pass"] == 1
    assert correctness["ragas"]["metrics"] == ["answer_correctness", "factual_correctness"]
    assert (correctness["ragas"]["pass"], correctness["ragas"]["fail"]) == (1, 1)
    assert "score" not in correctness["ragas"], "scores must not be averaged across metrics"
    assert ("retrieval", None) in groups, "unmapped dimensions are kept, not dropped or guessed"


# ---------- disagreements ----------

def test_disagreement_detected_on_same_case_and_dimension():
    rows = [to_row(_entry(r)) for r in (
        _record(framework="custom", case="A", status="PASS"),
        _record(framework="ragas", metric="answer_correctness", case="A", status="FAIL"),
        _record(framework="deepeval", metric="geval_answer_correctness", case="A", status="PASS"),
    )]

    [item] = find_disagreements(rows)

    assert item["case_id"] == "llm_correctness.json::A"
    assert set(item["frameworks"]) == {"custom", "ragas", "deepeval"}


@pytest.mark.parametrize("other", [
    dict(framework="ragas", metric="answer_correctness", case="A", status="PASS"),  # agreement
    dict(framework="ragas", metric="answer_correctness", case="A", status="BASELINE"),  # no verdict
    dict(framework="ragas", metric="answer_correctness", case="B", status="FAIL"),  # different case
    dict(framework="ragas", metric="answer_correctness", case="A", status="FAIL", quality_dimension="relevancy"),
    dict(framework="custom", metric="other_metric", case="A", status="FAIL"),  # same framework only
])
def test_no_disagreement_without_differing_verdicts_across_frameworks(other):
    rows = [to_row(_entry(_record(framework="custom", case="A", status="PASS"))), to_row(_entry(_record(**other)))]
    assert find_disagreements(rows) == []


# ---------- end to end ----------

def test_build_dashboard_current_scope_marks_disagreeing_rows(tmp_path):
    latest = tmp_path / "latest.json"
    _write_latest(latest, [
        _entry(_record(framework="custom", case="A", status="PASS")),
        _entry(_record(framework="ragas", metric="answer_correctness", case="A", status="FAIL", run_id="r")),
        _entry(_record(framework="custom", metric="judge_verdict", run_id="old", completed_at="2026-09-01T00:00:00+00:00")),
        "malformed",
    ])

    current = build_dashboard(latest, tmp_path / "events.jsonl", "current")
    everything = build_dashboard(latest, tmp_path / "events.jsonl", "all")

    assert current["summary"]["overall"]["total"] == 2
    assert everything["summary"]["overall"]["total"] == 3
    assert all(r["disagreement"] for r in current["rows"])
    assert len(current["warnings"]) == 1
    assert {r["run_id"] for r in current["runs"]} == {"run-1", "r"}


def test_build_dashboard_rejects_unknown_scope(tmp_path):
    with pytest.raises(ValueError):
        build_dashboard(tmp_path / "latest.json", tmp_path / "events.jsonl", "bogus")
