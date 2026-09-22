import json
from dataclasses import dataclass, field, asdict
from typing import Optional


class Status:
    """Valid values for EvaluationResult.status.

    PASS / FAIL: score computed, a threshold was configured, and the gate
        was evaluated against it.
    BASELINE: score computed, but no threshold is configured yet.
    N/A: the evaluation ran but produced no meaningful measurement (e.g.
        RAGAS Faithfulness returning NaN for a zero-statement response) —
        distinct from a failed pipeline.
    ERROR: the evaluation pipeline itself failed (exception, connection
        error, etc.) — no reliable score; see error_message.
    """

    PASS = "PASS"
    FAIL = "FAIL"
    BASELINE = "BASELINE"
    NOT_APPLICABLE = "N/A"
    ERROR = "ERROR"

    ALL = frozenset({PASS, FAIL, BASELINE, NOT_APPLICABLE, ERROR})
    # Statuses that carry a meaningful numeric measurement, as opposed to a
    # failed/non-informative attempt. Used by aggregate.py to decide
    # latest_valid vs latest_attempt.
    VALID_MEASUREMENT = frozenset({PASS, FAIL, BASELINE})


@dataclass
class EvaluationResult:
    # Identity
    evaluation_id: str
    run_id: str
    completed_at: str  # ISO8601

    # Universal
    framework: str  # "custom" | "ragas" | "deepeval"
    phase: str  # "retrieval" | "generation"
    suite: str
    case_id: str
    case_name: str
    metric: str
    status: str  # one of Status.ALL
    query: str

    quality_dimension: Optional[str] = None
    score: Optional[float] = None
    threshold: Optional[float] = None
    error_message: Optional[str] = None
    duration_ms: Optional[float] = None
    metadata: dict = field(default_factory=dict)

    # Retrieval-only
    retrieved_contexts: Optional[list] = None
    expected_ids: Optional[list] = None
    retrieved_ids: Optional[list] = None
    filter: Optional[dict] = None

    # Generation-only
    context: Optional[list] = None
    assistant_answer: Optional[str] = None
    reference_answer: Optional[str] = None
    expected_confidence: Optional[bool] = None
    actual_confidence: Optional[bool] = None
    judge_reason: Optional[str] = None

    # Model identity
    assistant_model: Optional[str] = None
    judge_model: Optional[str] = None
    embedding_model: Optional[str] = None

    # Resource counters.
    # assistant_calls is reserved for real, measured Assistant call
    # telemetry (fresh generation vs. cache reuse), but that tracking was
    # deliberately removed from the RAGAS Generation tests to keep them
    # simple — none of them currently populate this field, so it is always
    # None today. Left optional rather than removed, in case a future
    # caller wants to report it without a schema change.
    assistant_calls: Optional[int] = None
    # estimated_judge_calls / estimated_embedding_calls are NOT measured at
    # runtime — they are derived from known metric implementation behavior
    # (e.g. RAGAS Faithfulness makes exactly 2 Judge calls per invocation,
    # confirmed by reading its source). The "estimated_" prefix is
    # deliberate: never rename these to judge_calls/embedding_calls, which
    # would misrepresent an estimate as measured telemetry. Leave both None
    # until either a real count is known, or real runtime instrumentation
    # exists.
    estimated_judge_calls: Optional[int] = None
    estimated_embedding_calls: Optional[int] = None

    def __post_init__(self):
        if self.status not in Status.ALL:
            raise ValueError(f"Invalid status {self.status!r}; must be one of {sorted(Status.ALL)}")

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json_line(self) -> str:
        return json.dumps(self.to_dict(), default=str)
