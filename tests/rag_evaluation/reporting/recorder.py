import uuid
from datetime import datetime, timezone
from pathlib import Path

from tests.rag_evaluation.reporting.schema import EvaluationResult


class Recorder:
    """Appends EvaluationResult records to a JSONL event log, one line per
    completed evaluation, flushed immediately.

    This is local single-user evaluation tooling, not a transactional
    database: each record is flush()-ed so a crash mid-batch loses at most
    the one in-flight record, never previously-completed ones, but we
    deliberately do not call os.fsync() per record — that guards against a
    power loss / OS crash scenario this tool doesn't need to survive, at a
    real per-call disk-sync cost that would slow down every batch.
    """

    def __init__(self, events_path: Path, run_id: str = None):
        self.events_path = Path(events_path)
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or uuid.uuid4().hex

    def record(self, **fields) -> EvaluationResult:
        fields.setdefault("evaluation_id", uuid.uuid4().hex)
        fields.setdefault("run_id", self.run_id)
        fields.setdefault("completed_at", datetime.now(timezone.utc).isoformat())

        result = EvaluationResult(**fields)

        with open(self.events_path, "a", encoding="utf-8") as f:
            f.write(result.to_json_line() + "\n")
            f.flush()

        return result
