"""Loading, run selection and summaries for the Evaluation Dashboard.

Pure functions over the existing reporting output — no new result format.

Run selection ("current" scope): every pytest session writes its own run_id,
so no single run covers every framework. latest.json keeps one entry per
(framework, phase, case_id, metric) forever, including metrics/cases that a
later run of the same suite no longer produces (e.g. retired Custom
Generation metrics, early RAGAS BASELINE sweeps). For each
(framework, phase, suite) the run that recorded that suite most recently is
treated as current, and only entries whose latest_attempt came from that run
are shown. The "all" scope shows every latest.json entry unfiltered.
"""
import json
import logging
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from reporting.aggregate import aggregate
from reporting.schema import Status

logger = logging.getLogger(__name__)

SCOPE_CURRENT = "current"
SCOPE_ALL = "all"
SCOPES = (SCOPE_CURRENT, SCOPE_ALL)

CORE_FIELDS = ("run_id", "framework", "phase", "suite", "case_id", "metric", "status")
GATED_STATUSES = (Status.PASS, Status.FAIL)

# Framework-native scores stored in metadata alongside the shared 0-1 score.
NATIVE_SCORE_KEYS = ("raw_rubric_score",)

# Custom suite-level ranking metrics are recorded under a synthetic aggregate
# case whose `query` is "[aggregate over N queries]" (see
# tests/rag_evaluation/frameworks/custom/retrieval/test_ranking_metrics.py).
AGGREGATE_QUERY_PREFIX = "[aggregate over "


class ReportLoadError(Exception):
    """Core report data exists but can't be read — shown as an error, never
    replaced with fabricated values."""


def load_report(latest_path: Path, events_path: Path) -> dict:
    """Returns {"source", "generated_at", "results"} or raises ReportLoadError.

    Prefers latest.json. If it doesn't exist yet but events.jsonl does, the
    existing aggregate() helper folds events.jsonl in memory (nothing is
    written). If neither exists, returns an empty report.
    """
    latest_path, events_path = Path(latest_path), Path(events_path)

    if latest_path.exists():
        try:
            with open(latest_path, "r", encoding="utf-8") as f:
                report = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            raise ReportLoadError(f"Could not read {latest_path.name}: {exc}") from exc
        if not isinstance(report, dict) or not isinstance(report.get("results"), list):
            raise ReportLoadError(f"{latest_path.name} has no 'results' list")
        return {"source": latest_path.name, "generated_at": report.get("generated_at"), "results": report["results"]}

    if events_path.exists():
        report = aggregate(events_path)
        return {
            "source": f"{events_path.name} (aggregated in memory; {latest_path.name} not found)",
            "generated_at": report["generated_at"],
            "results": report["results"],
        }

    return {"source": None, "generated_at": None, "results": []}


def _completed_at(record: dict) -> datetime:
    try:
        return datetime.fromisoformat(record.get("completed_at"))
    except (TypeError, ValueError):
        return datetime.min.replace(tzinfo=timezone.utc)


def _malformed_reason(entry) -> "str | None":
    if not isinstance(entry, dict):
        return "entry is not an object"
    attempt = entry.get("latest_attempt")
    if not isinstance(attempt, dict):
        return "missing latest_attempt"
    missing = [f for f in CORE_FIELDS if not attempt.get(f)]
    if missing:
        return f"latest_attempt missing {', '.join(missing)}"
    if attempt["status"] not in Status.ALL:
        return f"unknown status {attempt['status']!r}"
    return None


def split_valid_entries(results: list) -> "tuple[list, list]":
    """Separates usable latest.json entries from malformed ones. One bad
    entry is reported as a warning, never allowed to break the dashboard."""
    valid, warnings = [], []
    for index, entry in enumerate(results):
        reason = _malformed_reason(entry)
        if reason:
            logger.warning("Skipping malformed report entry #%d: %s", index, reason)
            warnings.append(f"Skipped malformed entry #{index}: {reason}")
        else:
            valid.append(entry)
    return valid, warnings


def select_current_runs(entries: list) -> "tuple[list, dict]":
    """Keeps only entries from the most recent run of each
    (framework, phase, suite). Returns (entries, {group: run info})."""
    newest: dict[tuple, dict] = {}
    for entry in entries:
        attempt = entry["latest_attempt"]
        group = (attempt["framework"], attempt["phase"], attempt["suite"])
        if group not in newest or _completed_at(attempt) > _completed_at(newest[group]):
            newest[group] = attempt

    current_run = {group: record["run_id"] for group, record in newest.items()}
    selected = [
        e for e in entries
        if e["latest_attempt"]["run_id"]
        == current_run[(e["latest_attempt"]["framework"], e["latest_attempt"]["phase"], e["latest_attempt"]["suite"])]
    ]
    return selected, current_run


def is_aggregate(record: dict) -> bool:
    return str(record.get("query") or "").startswith(AGGREGATE_QUERY_PREFIX)


def native_score(record: dict) -> "dict | None":
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    for key in NATIVE_SCORE_KEYS:
        if metadata.get(key) is not None:
            return {"name": key, "value": metadata[key]}
    return None


def to_row(entry: dict) -> dict:
    """Flattens a latest.json entry into a dashboard row: the latest_attempt
    record as-is, plus a few derived presentation fields."""
    attempt = entry["latest_attempt"]
    row = dict(attempt)
    row["metadata"] = attempt.get("metadata") if isinstance(attempt.get("metadata"), dict) else {}
    row["is_aggregate"] = is_aggregate(attempt)
    row["native_score"] = native_score(attempt)

    valid = entry.get("latest_valid")
    # When the latest attempt was ERROR/N/A, surface the last real measurement
    # separately rather than substituting it for the current status.
    if isinstance(valid, dict) and valid.get("evaluation_id") != attempt.get("evaluation_id"):
        row["previous_valid"] = {k: valid.get(k) for k in ("status", "score", "threshold", "run_id", "completed_at")}
    else:
        row["previous_valid"] = None
    return row


def pass_rate(pass_count: int, fail_count: int) -> "float | None":
    """PASS / (PASS + FAIL). BASELINE / N/A / ERROR have no gate verdict and
    are excluded; None when nothing was gated."""
    gated = pass_count + fail_count
    return pass_count / gated if gated else None


def _counts(rows: list) -> dict:
    statuses = Counter(r["status"] for r in rows)
    passed, failed = statuses.get(Status.PASS, 0), statuses.get(Status.FAIL, 0)
    return {
        "total": len(rows),
        "pass": passed,
        "fail": failed,
        "other": len(rows) - passed - failed,
        "by_status": dict(statuses),
        "pass_rate": pass_rate(passed, failed),
    }


def summarize(rows: list) -> dict:
    overall = _counts(rows)
    overall["frameworks"] = sorted({r["framework"] for r in rows})
    overall["by_phase"] = dict(Counter(r["phase"] for r in rows))

    by_framework = defaultdict(list)
    for r in rows:
        by_framework[r["framework"]].append(r)
    frameworks = {fw: _counts(fw_rows) for fw, fw_rows in sorted(by_framework.items())}
    return {"overall": overall, "frameworks": frameworks}


def group_by_dimension(rows: list) -> list:
    """Per (phase, quality_dimension): counts and metric names per framework.
    Scores from different frameworks are never averaged together."""
    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        groups[(r["phase"], r.get("quality_dimension"))][r["framework"]].append(r)

    out = []
    for (phase, dimension), per_fw in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1] or "~")):
        out.append({
            "phase": phase,
            "quality_dimension": dimension,
            "frameworks": {
                fw: {**_counts(fw_rows), "metrics": sorted({r["metric"] for r in fw_rows})}
                for fw, fw_rows in sorted(per_fw.items())
            },
        })
    return out


def find_disagreements(rows: list) -> list:
    """Cases where 2+ frameworks gave a PASS/FAIL verdict for the same
    (phase, case_id, quality_dimension) and those verdicts are not all equal.
    Matching uses the stored case_id (golden file + case name) exactly — no
    fuzzy matching. BASELINE/N/A/ERROR carry no verdict and are ignored."""
    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["status"] in GATED_STATUSES and r.get("quality_dimension") and not r["is_aggregate"]:
            groups[(r["phase"], r["case_id"], r["quality_dimension"])][r["framework"]].append(r)

    out = []
    for (phase, case_id, dimension), per_fw in sorted(groups.items()):
        statuses = {r["status"] for fw_rows in per_fw.values() for r in fw_rows}
        if len(per_fw) >= 2 and len(statuses) > 1:
            out.append({
                "phase": phase,
                "case_id": case_id,
                "quality_dimension": dimension,
                "frameworks": {
                    fw: [{"metric": r["metric"], "status": r["status"], "evaluation_id": r["evaluation_id"]} for r in fw_rows]
                    for fw, fw_rows in sorted(per_fw.items())
                },
            })
    return out


def build_dashboard(latest_path: Path, events_path: Path, scope: str = SCOPE_CURRENT) -> dict:
    """Everything the dashboard page needs, in one payload."""
    if scope not in SCOPES:
        raise ValueError(f"Unknown scope {scope!r}; expected one of {SCOPES}")

    report = load_report(latest_path, events_path)
    entries, warnings = split_valid_entries(report["results"])

    runs = []
    if scope == SCOPE_CURRENT:
        entries, current_run = select_current_runs(entries)
        for (framework, phase, suite), run_id in sorted(current_run.items()):
            runs.append({"framework": framework, "phase": phase, "suite": suite, "run_id": run_id})

    rows = [to_row(e) for e in entries]
    rows.sort(key=lambda r: (r["framework"], r["phase"], r.get("quality_dimension") or "~", r["case_id"], r["metric"]))

    disagreements = find_disagreements(rows)
    disagreeing_ids = {
        m["evaluation_id"] for d in disagreements for fw_rows in d["frameworks"].values() for m in fw_rows
    }
    for r in rows:
        r["disagreement"] = r.get("evaluation_id") in disagreeing_ids

    return {
        "scope": scope,
        "source": report["source"],
        "generated_at": report["generated_at"],
        "warnings": warnings,
        "runs": runs,
        "summary": summarize(rows),
        "dimensions": group_by_dimension(rows),
        "disagreements": disagreements,
        "rows": rows,
    }
