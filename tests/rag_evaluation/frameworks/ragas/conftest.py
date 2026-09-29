import pytest
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from config import LLM_BASE_URL, LLM_API_KEY, JUDGE_MODEL


@pytest.fixture(scope="session")
def ragas_judge_llm():
    """Shared RAGAS evaluator LLM, wired to the local Judge model/endpoint.
    Used by both Generation and Retrieval metrics."""
    client = AsyncOpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    return llm_factory(
        model=JUDGE_MODEL,
        provider="openai",
        client=client,
        # Disables Qwen3's "thinking" mode, matching LLMService's judge path.
        extra_body={"reasoning_effort": "none"},
    )


def resolve_min_score(framework_config, metric_name):
    """Resolves one metric's calibrated threshold from an evaluation.ragas
    config block. min_score is a single number when one metric owns the
    dataset, or a {metric_name: number} object when two RAGAS metrics
    share it with independent thresholds."""
    min_score = framework_config.get("min_score")
    if isinstance(min_score, dict):
        return min_score.get(metric_name)
    return min_score
