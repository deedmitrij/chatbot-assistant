import json
from datetime import datetime, timezone
from pathlib import Path

from tests.rag_evaluation.reporting.schema import Status

KEY_FIELDS = ("framework", "phase", "case_id", "metric")


def _parse_completed_at(record: dict) -> datetime:
    value = record.get("completed_at")
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        # Missing/unparseable timestamp — treat as earliest possible so a
        # malformed-but-JSON-valid record never wins over a well-formed one.
        return datetime.min.replace(tzinfo=timezone.utc)


def _read_events(events_path: Path):
    """Yields parsed JSON records from a JSONL file, skipping any line that
    fails to parse (e.g. a truncated final line from a crashed run) rather
    than letting one bad line make the whole file unreadable."""
    events_path = Path(events_path)
    if not events_path.exists():
        return
    with open(events_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def aggregate(events_path: Path) -> dict:
    """Folds the JSONL event log into one entry per (framework, phase,
    case_id, metric), tracking two pointers per key:

    - latest_attempt: the most recent record regardless of status, so a
      fresh ERROR is visible (e.g. in Case Explorer/Findings).
    - latest_valid: the most recent record whose status carries a real
      measurement (PASS/FAIL/BASELINE) — a newer ERROR or N/A never
      overwrites this, so the last real score keeps showing on the
      dashboard even if the most recent attempt failed.
    """
    groups: dict[tuple, dict] = {}

    for record in _read_events(events_path):
        if not all(field in record for field in KEY_FIELDS):
            continue
        key = tuple(record[field] for field in KEY_FIELDS)
        entry = groups.setdefault(key, {"latest_attempt": None, "latest_valid": None})

        completed_at = _parse_completed_at(record)

        if entry["latest_attempt"] is None or completed_at >= _parse_completed_at(entry["latest_attempt"]):
            entry["latest_attempt"] = record

        if record.get("status") in Status.VALID_MEASUREMENT:
            if entry["latest_valid"] is None or completed_at >= _parse_completed_at(entry["latest_valid"]):
                entry["latest_valid"] = record

    results = [
        {
            "key": dict(zip(KEY_FIELDS, key)),
            "latest_attempt": entry["latest_attempt"],
            "latest_valid": entry["latest_valid"],
        }
        for key, entry in groups.items()
    ]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }


def write_latest(events_path: Path, output_path: Path) -> dict:
    aggregated = aggregate(events_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(aggregated, f, indent=2, default=str)
    return aggregated
