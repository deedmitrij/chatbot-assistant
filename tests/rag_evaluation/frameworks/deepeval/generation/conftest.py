import json
import os
import pytest
from deepeval.models import OllamaModel
from config import PROJECT_ROOT, RAG_EVALUATION_DATA_DIR, CHAT_MODEL, JUDGE_MODEL
from backend.constants import LLMRole
from backend.services.llm.llm_service import LLMService
from tests.rag_evaluation.reporting.recorder import Recorder
from tests.rag_evaluation.reporting.aggregate import write_latest
from tests.rag_evaluation.reporting.dimensions import get_quality_dimension
from tests.rag_evaluation.reporting.schema import Status

EVENTS_PATH = PROJECT_ROOT / "reports" / "results" / "events.jsonl"
LATEST_PATH = PROJECT_ROOT / "reports" / "results" / "latest.json"

# The default 88.5s per-attempt timeout is too short for qwen3:8b's
# thinking-mode latency: the native OllamaModel integration has no way to
# disable thinking mode (unlike LLMService's extra_body={"reasoning_effort":
# "none"}), confirmed in the DeepEval Judge smoke test and reused as-is for
# Retrieval (see ../retrieval/conftest.py). Same override, scoped to this
# framework's Generation tests only.
os.environ.setdefault("DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE", "240")

GENERATION_FILES = (
    "llm_faithfulness.json",
    "llm_correctness.json",
    "llm_hallucination.json",
    "llm_relevancy.json",
    "llm_negative_constraint.json",
    "llm_brand_consistency.json",
)


@pytest.fixture(scope="session")
def assistant_llm_service():
    """Real Assistant / System Under Test, exactly the production generation
    path -- same fixture definition as RAGAS's own (see
    ../../ragas/generation/conftest.py), independently constructed rather
    than imported, matching the existing per-framework convention."""
    return LLMService(role=LLMRole.ASSISTANT)


@pytest.fixture(scope="session")
def deepeval_judge_model():
    """Native DeepEval Ollama Judge, wired to the same local Judge model as
    Retrieval's DeepEval suite (see ../retrieval/conftest.py) and as
    Custom/RAGAS (JUDGE_MODEL, qwen3:8b). Project-scoped construction -- no
    `deepeval set-ollama` global config."""
    return OllamaModel(
        model=JUDGE_MODEL,
        base_url="http://localhost:11434",
        temperature=0,
    )


@pytest.fixture(scope="session")
def assistant_response_cache():
    """Session-scoped cache so each case's real Assistant response is
    generated once and reused across every DeepEval Generation metric that
    applies to it, keyed by the globally-stable case_id (source file + case
    name). Same pattern and same rationale as RAGAS's own
    assistant_response_cache (see ../../ragas/generation/conftest.py) --
    kept as its own session-scoped dict rather than shared across
    frameworks, so a DeepEval-only pytest run never depends on RAGAS's
    fixtures being collected."""
    return {}


@pytest.fixture(scope="session")
def recorder():
    """One Recorder per pytest session, persisting to the same events.jsonl
    the reporting subsystem's aggregator reads from -- same pattern as
    Custom's, RAGAS's and DeepEval Retrieval's own conftest.py files."""
    return Recorder(EVENTS_PATH)


def get_or_generate_response(cache, llm_service, case):
    """Generate the real Assistant result for a case once, memoized by
    case_id, and reused across every DeepEval Generation metric that applies
    to that case. Exact port of RAGAS's own helper (see
    ../../ragas/generation/conftest.py).

    Returns the full {"answer": str, "confidence": bool} dict
    LLMService.get_answer() already returns, cached as-is so
    actual_confidence can be recorded honestly alongside the answer text.
    """
    case_id = case["case_id"]
    if case_id not in cache:
        cache[case_id] = llm_service.get_answer(query=case["query"], context=case["context"])
    return cache[case_id]


def validate_deepeval_metric_config(case_id, metric_name, config):
    """Enforces the explicit applicability/skip schema for one
    evaluation.deepeval.<metric_name> config. Same contract as RAGAS's
    validate_ragas_metric_config (see ../../ragas/generation/conftest.py):

    - skip=true must carry a non-empty (non-whitespace) skip_reason.
    - skip=true must NOT also carry a min_score (a skipped metric has no
      threshold to speak of).
    """
    if not config.get("skip"):
        return
    skip_reason = config.get("skip_reason")
    if not isinstance(skip_reason, str) or not skip_reason.strip():
        raise ValueError(
            f"{case_id}: evaluation.deepeval.{metric_name} has skip=true but "
            f"skip_reason is missing or empty"
        )
    if "min_score" in config:
        raise ValueError(
            f"{case_id}: evaluation.deepeval.{metric_name} has skip=true but "
            f"also declares min_score; skipped metrics must not carry a threshold"
        )


def get_generation_deepeval_cases(metric_name):
    """Load DeepEval-eligible Generation cases for one metric.

    Applicability is read explicitly from evaluation.deepeval.<metric_name>,
    never hardcoded by case name -- same contract as RAGAS's
    get_generation_ragas_cases (see ../../ragas/generation/conftest.py):

    - metric key absent -> not configured for this case, ignored.
    - metric key present with skip=true -> intentionally not applicable
      (validated via validate_deepeval_metric_config), ignored.
    - metric key present otherwise -> applicable, included.
    """
    test_cases = []
    for file_name in GENERATION_FILES:
        file_path = RAG_EVALUATION_DATA_DIR / "generation" / file_name
        with open(file_path, "r") as f:
            suites = json.load(f)
        for suite in suites:
            context = suite["context"]
            for case in suite["cases"]:
                case_id = f"{file_name}::{case['name']}"
                metric_config = case.get("evaluation", {}).get("deepeval", {}).get(metric_name)
                if metric_config is None:
                    continue
                validate_deepeval_metric_config(case_id, metric_name, metric_config)
                if metric_config.get("skip"):
                    continue
                case["context"] = context
                case["case_id"] = case_id
                test_cases.append(case)
    return test_cases


def record_generation_result(*, recorder, case, metric_name, score, reason, assistant_result):
    """Reporting-only: persists one EvaluationResult for an already-measured
    DeepEval metric's score/reason, then re-materializes latest.json. Mirrors
    RAGAS's record_generation_result (see ../../ragas/generation/conftest.py),
    adapted for DeepEval's plain (score, reason) attributes instead of a
    RAGAS MetricResult object -- never constructs or measures the metric.

    Status is derived generically from the case's own
    evaluation.deepeval[metric_name] config:
    - NaN score -> status=N/A, score=None, threshold=None.
    - finite value, min_score configured -> status=PASS/FAIL, threshold=min_score.
    - finite value, no min_score configured yet -> status=BASELINE, threshold=None.
    """
    is_nan = score != score
    min_score = case["evaluation"]["deepeval"][metric_name].get("min_score")

    if is_nan:
        status = Status.NOT_APPLICABLE
        recorded_score = None
        threshold = None
    else:
        recorded_score = float(score)
        if min_score is None:
            status = Status.BASELINE
            threshold = None
        else:
            status = Status.PASS if recorded_score >= min_score else Status.FAIL
            threshold = min_score

    recorder.record(
        framework="deepeval",
        phase="generation",
        suite=case["case_id"].split("::", 1)[0],
        case_id=case["case_id"],
        case_name=case["name"],
        metric=metric_name,
        quality_dimension=get_quality_dimension("deepeval", metric_name),
        query=case["query"],
        threshold=threshold,
        status=status,
        score=recorded_score,
        context=case["context"],
        assistant_answer=assistant_result["answer"],
        reference_answer=case.get("reference_answer"),
        expected_confidence=case.get("expected_confidence"),
        actual_confidence=assistant_result["confidence"],
        judge_reason=reason,
        assistant_model=CHAT_MODEL,
        judge_model=JUDGE_MODEL,
        embedding_model=None,
    )
    write_latest(EVENTS_PATH, LATEST_PATH)
