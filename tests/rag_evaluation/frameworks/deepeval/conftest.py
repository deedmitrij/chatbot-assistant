import os
import pytest
from deepeval.models import OllamaModel
from config import JUDGE_MODEL


# The default 88.5s per-attempt timeout is too short for qwen3:8b's
# thinking-mode latency: the native OllamaModel integration has no way to
# disable thinking mode (unlike LLMService's extra_body={"reasoning_effort":
# "none"}). Applies to both Generation and Retrieval Judge calls.
os.environ.setdefault("DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE", "240")


@pytest.fixture(scope="session")
def deepeval_judge_model():
    """Shared native DeepEval Ollama Judge (JUDGE_MODEL, qwen3:8b).
    Used by both Generation and Retrieval metrics."""
    return OllamaModel(
        model=JUDGE_MODEL,
        base_url="http://localhost:11434",
        temperature=0,
    )


def resolve_min_score(framework_config, metric_name):
    """Resolves one metric's calibrated threshold from an evaluation.deepeval
    config block. min_score is a single number when one metric owns the
    dataset, or a {metric_name: number} object when two DeepEval metrics
    share it with independent thresholds."""
    min_score = framework_config.get("min_score")
    if isinstance(min_score, dict):
        return min_score.get(metric_name)
    return min_score
