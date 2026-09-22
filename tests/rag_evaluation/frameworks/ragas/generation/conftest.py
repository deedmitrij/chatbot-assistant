import json
import pytest
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.embeddings.huggingface_provider import HuggingFaceEmbeddings
from config import (
    PROJECT_ROOT,
    RAG_EVALUATION_DATA_DIR,
    LLM_BASE_URL,
    LLM_API_KEY,
    CHAT_MODEL,
    JUDGE_MODEL,
    EMBEDDING_MODEL,
)
from backend.constants import LLMRole
from backend.services.llm.llm_service import LLMService
from tests.rag_evaluation.reporting.recorder import Recorder
from tests.rag_evaluation.reporting.aggregate import write_latest
from tests.rag_evaluation.reporting.dimensions import get_quality_dimension
from tests.rag_evaluation.reporting.schema import Status

EVENTS_PATH = PROJECT_ROOT / "reports" / "results" / "events.jsonl"
LATEST_PATH = PROJECT_ROOT / "reports" / "results" / "latest.json"


GENERATION_FILES = (
    "llm_faithfulness.json",
    "llm_correctness.json",
    "llm_hallucination.json",
    "llm_relevancy.json",
    "llm_negative_constraint.json",
    "llm_brand_consistency.json",
)

# Project choice, not a reproduction of RAGAS's benchmark defaults: this is a
# local AI Quality Engineering lab comparing evaluation frameworks, not an
# attempt to match RAGAS's benchmark-grade settings. AnswerRelevancy's
# default strictness=3 means 3 Judge calls per case; strictness=1 keeps the
# metric's semantics (paraphrase generation + cosine similarity against the
# original query, noncommittal-answer detection) while cutting the full
# Generation baseline's Judge-call count roughly in half.
ANSWER_RELEVANCY_STRICTNESS = 1


@pytest.fixture(scope="session")
def assistant_llm_service():
    """Real Assistant / System Under Test, exactly the production generation path."""
    return LLMService(role=LLMRole.ASSISTANT)


@pytest.fixture(scope="session")
def ragas_judge_llm():
    """RAGAS evaluator LLM, wired to the same local Judge model/endpoint as
    Retrieval's RAGAS suite (see ../retrieval/conftest.py)."""
    client = AsyncOpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    return llm_factory(
        model=JUDGE_MODEL,
        provider="openai",
        client=client,
        # Disables Qwen3's "thinking" mode, matching LLMService's judge path.
        extra_body={"reasoning_effort": "none"},
    )


@pytest.fixture(scope="session")
def ragas_embeddings():
    """Local, offline embeddings for AnswerRelevancy/AnswerCorrectness.

    Reuses the project's existing EMBEDDING_MODEL, run locally via
    sentence-transformers (use_api=False) rather than the HuggingFace hosted
    Inference API — no network/token dependency, consistent with the local
    Ollama-first setup used for chat and judging.
    """
    return HuggingFaceEmbeddings(model=EMBEDDING_MODEL, use_api=False)


@pytest.fixture(scope="session")
def assistant_response_cache():
    """Session-scoped cache so each case's real Assistant response is
    generated once and reused across every RAGAS metric that applies to it,
    keyed by the globally-stable case_id (source file + case name) rather
    than the bare case name, since case names aren't guaranteed unique
    across the six generation golden-data files."""
    return {}


@pytest.fixture(scope="session")
def recorder():
    """One Recorder per pytest session, so every metric module invoked in
    the same run shares one run_id. Persists to the same events.jsonl the
    reporting subsystem's aggregator reads from."""
    return Recorder(EVENTS_PATH)


def get_or_generate_response(cache, llm_service, case):
    """Generate the real Assistant result for a case once, memoized by
    case_id, and reused across every RAGAS metric that applies to that case.

    Returns the full {"answer": str, "confidence": bool} dict
    LLMService.get_answer() already returns, cached as-is so
    actual_confidence can be recorded honestly alongside the answer text,
    not just the answer string.
    """
    case_id = case["case_id"]
    if case_id not in cache:
        cache[case_id] = llm_service.get_answer(query=case["query"], context=case["context"])
    return cache[case_id]


def record_generation_result(*, recorder, case, metric_name, result, assistant_result, embedding_model=None):
    """Reporting-only: persists one EvaluationResult for an already-computed
    RAGAS MetricResult, then re-materializes latest.json.

    Takes the finished `result` as input — it never constructs a metric,
    never calls .score(...), and never alters the result.

    Status is derived generically from the case's own
    evaluation.ragas[metric_name] config, not from anything metric-specific
    here, so this already supports thresholds for every RAGAS Generation
    metric as each gets calibrated, not just Faithfulness:
    - NaN -> status=N/A, score=None, threshold=None.
    - finite value, min_score configured -> status=PASS/FAIL based on
      score >= min_score, threshold=min_score.
    - finite value, no min_score configured yet -> status=BASELINE,
      threshold=None (unchanged from before calibration).

    embedding_model is persisted exactly as passed by the caller — this
    helper does not infer it from metric_name, so callers whose metric
    doesn't use embeddings simply omit the argument (stays None).
    """
    is_nan = result.value != result.value
    min_score = case["evaluation"]["ragas"][metric_name].get("min_score")

    if is_nan:
        status = Status.NOT_APPLICABLE
        score = None
        threshold = None
    else:
        score = float(result.value)
        if min_score is None:
            status = Status.BASELINE
            threshold = None
        else:
            status = Status.PASS if score >= min_score else Status.FAIL
            threshold = min_score

    recorder.record(
        framework="ragas",
        phase="generation",
        suite=case["case_id"].split("::", 1)[0],
        case_id=case["case_id"],
        case_name=case["name"],
        metric=metric_name,
        quality_dimension=get_quality_dimension("ragas", metric_name),
        query=case["query"],
        threshold=threshold,
        status=status,
        score=score,
        context=case["context"],
        assistant_answer=assistant_result["answer"],
        reference_answer=case.get("reference_answer"),
        expected_confidence=case.get("expected_confidence"),
        actual_confidence=assistant_result["confidence"],
        judge_reason=getattr(result, "reason", None),
        assistant_model=CHAT_MODEL,
        judge_model=JUDGE_MODEL,
        embedding_model=embedding_model,
    )
    write_latest(EVENTS_PATH, LATEST_PATH)


def validate_ragas_metric_config(case_id, metric_name, config):
    """Enforces the explicit applicability/skip schema for one
    evaluation.ragas.<metric_name> config:

    - skip=true must carry a non-empty (non-whitespace) skip_reason.
    - skip=true must NOT also carry a min_score (a skipped metric has no
      threshold to speak of).

    Applicable configs (no skip key) aren't validated here beyond that —
    each test module's own contract assertions cover the shape it needs.
    """
    if not config.get("skip"):
        return
    skip_reason = config.get("skip_reason")
    if not isinstance(skip_reason, str) or not skip_reason.strip():
        raise ValueError(
            f"{case_id}: evaluation.ragas.{metric_name} has skip=true but "
            f"skip_reason is missing or empty"
        )
    if "min_score" in config:
        raise ValueError(
            f"{case_id}: evaluation.ragas.{metric_name} has skip=true but "
            f"also declares min_score; skipped metrics must not carry a threshold"
        )


def get_generation_ragas_cases(metric_name):
    """Load RAGAS-eligible Generation cases for one metric.

    Applicability is read explicitly from evaluation.ragas.<metric_name>,
    never hardcoded by case name here:

    - metric key absent -> not configured for this case, ignored.
    - metric key present with skip=true -> intentionally not applicable
      (validated via validate_ragas_metric_config), ignored — no test node
      is generated and nothing is recorded for it.
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
                metric_config = case.get("evaluation", {}).get("ragas", {}).get(metric_name)
                if metric_config is None:
                    continue
                validate_ragas_metric_config(case_id, metric_name, metric_config)
                if metric_config.get("skip"):
                    continue
                case["context"] = context
                case["case_id"] = case_id
                test_cases.append(case)
    return test_cases
