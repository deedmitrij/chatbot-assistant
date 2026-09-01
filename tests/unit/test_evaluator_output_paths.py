from pathlib import Path

import pandas as pd

from backend.evaluation import evaluator as evaluator_module


EVALUATION_DIR = Path(evaluator_module.__file__).resolve().parent


class FakeChatManagerForEval:
    """Duck-typed stand-in for ChatManager. No real LLM/vector DB/Telegram/SQLite involved."""

    def __init__(self, results_by_question):
        self._results_by_question = results_by_question

    def process_message_for_eval(self, query):
        return self._results_by_question[query]


class FakeRagasResult:
    """Stand-in for ragas.evaluate()'s return value; only .to_pandas() is used by run()."""

    def __init__(self, df):
        self._df = df

    def to_pandas(self):
        return self._df


def _make_evaluator(chat_manager):
    """
    Build a RAGEvaluator without running RAGEvaluator.__init__, which would
    otherwise construct real AsyncOpenAI/HuggingFaceEmbeddings clients (network
    calls / model downloads). run() only needs chat_manager, metrics, llm, and
    embeddings to exist as attributes; metrics/llm/embeddings are never
    inspected because ragas.evaluate() itself is monkeypatched below.
    """
    evaluator = evaluator_module.RAGEvaluator.__new__(evaluator_module.RAGEvaluator)
    evaluator.chat_manager = chat_manager
    evaluator.metrics = []
    evaluator.llm = None
    evaluator.embeddings = None
    return evaluator


def test_bundled_output_paths_resolve_next_to_module_not_cwd(monkeypatch, tmp_path):
    """
    Regression test: evaluator.py's own results directory and CSV output
    paths must resolve relative to the evaluator module's own location, not
    the process's current working directory, so
    `python backend/evaluation/evaluator.py` saves results into
    backend/evaluation/results/ regardless of the caller's cwd.

    Today, the only references to these output paths are bare relative string
    literals (run()'s default parameter value and the two __main__ calls),
    which expose no cwd-independent path anywhere in the module — that is the
    root cause of the bug. This test targets the absolute, __file__-derived
    constants the fix is expected to introduce (RESULTS_DIR /
    EVAL_RESULTS_CSV_PATH / EVAL_SAFETY_RESULTS_CSV_PATH); pre-fix it fails
    with AttributeError because no such mechanism exists yet.
    """
    # Simulate running the script from an unrelated working directory, matching
    # `python backend/evaluation/evaluator.py` invoked from the repo root.
    monkeypatch.chdir(tmp_path)

    results_dir = evaluator_module.RESULTS_DIR
    results_csv = evaluator_module.EVAL_RESULTS_CSV_PATH
    safety_csv = evaluator_module.EVAL_SAFETY_RESULTS_CSV_PATH

    assert results_dir.is_absolute(), "Results directory must not be a bare cwd-relative string"
    assert results_csv.is_absolute(), "Results CSV path must not be a bare cwd-relative string"
    assert safety_csv.is_absolute(), "Safety CSV path must not be a bare cwd-relative string"
    assert results_dir == EVALUATION_DIR / "results"
    assert results_csv == results_dir / "evaluation_results.csv"
    assert safety_csv == results_dir / "evaluation_safety.csv"


def test_run_writes_output_to_given_path_and_creates_missing_parent_directory(monkeypatch, tmp_path):
    """
    Regression test: run() must save its results to exactly the (possibly
    absolute) output_path it's given, creating any missing parent directory
    first, instead of assuming the directory already exists (which is not
    guaranteed: backend/evaluation/results/ is gitignored, so it may not
    exist at all on a fresh clone).

    No real LLM/RAGAS/Hugging Face/ChromaDB/network calls are made:
    RAGEvaluator is built via __new__ (bypassing __init__), load_evaluation_data
    and ragas.evaluate are both replaced with in-memory fakes.
    """
    monkeypatch.chdir(tmp_path)

    fake_chat_manager = FakeChatManagerForEval({
        "When can I check in?": {"answer": "Check-in is at 3:00 PM.", "context": ["doc1"]},
    })
    evaluator = _make_evaluator(fake_chat_manager)
    monkeypatch.setattr(
        evaluator,
        "load_evaluation_data",
        lambda file_path: [{"question": "When can I check in?", "ground_truth": "Check-in is at 3:00 PM."}],
    )

    fake_df = pd.DataFrame([{"question": "When can I check in?", "answer": "Check-in is at 3:00 PM."}])
    monkeypatch.setattr(evaluator_module, "evaluate", lambda **kwargs: FakeRagasResult(fake_df))

    # A path whose parent directory does not exist yet, outside the simulated
    # cwd, to prove run() doesn't depend on cwd and safely creates missing dirs.
    output_path = tmp_path / "somewhere" / "nested" / "evaluation_results.csv"
    assert not output_path.parent.exists()

    result_df = evaluator.run("unused-dataset-path.json", output_path=str(output_path))

    assert output_path.exists(), "run() must create any missing parent directory before writing"
    assert result_df.equals(fake_df)
