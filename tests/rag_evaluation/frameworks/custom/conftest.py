from pathlib import Path
from config import CHAT_MODEL, JUDGE_MODEL, LLM_BASE_URL

# A single shared conftest here (rather than duplicate pytest_html_report_title
# /pytest_metadata implementations in the retrieval and generation conftest.py
# files) avoids two conflicting hook implementations firing when both suites
# are collected in one run.
#
# Both pytest-metadata's pytest_metadata hook and pytest-html's
# pytest_html_report_title hook fire before collection (from pytest_configure
# and pytest_sessionstart respectively), so which suite(s) are active can't be
# read from collected items — it's detected instead from the CLI path
# arguments (config.args), which pytest_configure already has access to.
_RETRIEVAL_DIR = (Path(__file__).parent / "retrieval").resolve()
_GENERATION_DIR = (Path(__file__).parent / "generation" / "nondeterministic").resolve()
_config_ref = {"config": None}


def _paths_overlap(a: Path, b: Path) -> bool:
    """True if a and b are the same directory, or one is an ancestor of the other."""
    return a == b or a in b.parents or b in a.parents


def _detect_active_suites(config) -> dict:
    arg_paths = [Path(str(a).split("::")[0]).resolve() for a in (config.args or [])]
    if not arg_paths:
        return {"retrieval": False, "generation": False}
    return {
        "retrieval": any(_paths_overlap(p, _RETRIEVAL_DIR) for p in arg_paths),
        "generation": any(_paths_overlap(p, _GENERATION_DIR) for p in arg_paths),
    }


def pytest_configure(config):
    """Stores the config reference so pytest_html_report_title (which isn't
    passed config) can detect the active suite(s) further below."""
    _config_ref["config"] = config


def pytest_html_report_title(report):
    """Sets the HTML report title (pytest-html) based on which suite(s) ran."""
    config = _config_ref["config"]
    if config is None:
        return
    suites = _detect_active_suites(config)

    if suites["retrieval"] and suites["generation"]:
        report.title = "Custom AI Evaluation Framework"
    elif suites["generation"]:
        report.title = "Custom LLM Generation Evaluation"
    elif suites["retrieval"]:
        report.title = "Custom RAG Retrieval Evaluation"


def pytest_metadata(metadata, config):
    """Adds report metadata rows shown in the HTML report's Environment table."""
    suites = _detect_active_suites(config)
    inference = "Local Ollama" if "localhost" in (LLM_BASE_URL or "") else LLM_BASE_URL

    if suites["retrieval"] and suites["generation"]:
        metadata["Evaluation type"] = "Custom pytest-based AI Evaluation"
        metadata["Retrieval"] = "ChromaDB / Top-K 3"
        metadata["Assistant model"] = CHAT_MODEL
        metadata["Judge model"] = JUDGE_MODEL
        metadata["Inference"] = inference
    elif suites["generation"]:
        metadata["Evaluation type"] = "Custom pytest-based LLM Generation"
        metadata["Assistant model"] = CHAT_MODEL
        metadata["Judge model"] = JUDGE_MODEL
        metadata["Inference"] = inference
    elif suites["retrieval"]:
        metadata["Evaluation type"] = "Custom pytest-based RAG Retrieval"
        metadata["Vector DB"] = "ChromaDB"
        metadata["Top-K"] = "3"
