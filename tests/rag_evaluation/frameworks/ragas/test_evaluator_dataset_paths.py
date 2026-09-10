from pathlib import Path

from backend.evaluation import evaluator as evaluator_module


EVALUATION_DIR = Path(evaluator_module.__file__).resolve().parent


def test_bundled_dataset_paths_resolve_next_to_module_not_cwd(monkeypatch, tmp_path):
    """
    Regression test: evaluator.py's own bundled dataset files (eval_dataset.json,
    eval_negative.json) must resolve relative to the evaluator module's own
    location, not the process's current working directory, so
    `python backend/evaluation/evaluator.py` behaves the same from any cwd
    (e.g. the repo root).

    Today, the only references to these filenames are bare relative string
    literals inside `if __name__ == "__main__":`, which is never executed on
    import and exposes no cwd-independent path anywhere in the module — that
    is the root cause of the bug. This test targets the absolute, __file__-
    derived constants the fix is expected to introduce (EVAL_DATASET_PATH /
    EVAL_NEGATIVE_DATASET_PATH); pre-fix it fails with AttributeError because
    no such cwd-independent mechanism exists yet anywhere in the module.

    No real LLM/Hugging Face/embeddings/ChromaDB/network calls are made:
    RAGEvaluator is only used via __new__ (bypassing __init__, which would
    otherwise construct real API clients), and only local file I/O is exercised.
    """
    # Simulate running the script from an unrelated working directory, matching
    # `python backend/evaluation/evaluator.py` invoked from the repo root.
    monkeypatch.chdir(tmp_path)

    dataset_path = evaluator_module.EVAL_DATASET_PATH
    negative_path = evaluator_module.EVAL_NEGATIVE_DATASET_PATH

    assert dataset_path.is_absolute(), "Dataset path must not be a bare cwd-relative string"
    assert negative_path.is_absolute(), "Dataset path must not be a bare cwd-relative string"
    assert dataset_path == EVALUATION_DIR / "eval_dataset.json"
    assert negative_path == EVALUATION_DIR / "eval_negative.json"

    # The actual point of the fix: the files must be found even though cwd is
    # now an unrelated temp directory.
    assert dataset_path.exists()
    assert negative_path.exists()

    evaluator = evaluator_module.RAGEvaluator.__new__(evaluator_module.RAGEvaluator)
    data = evaluator.load_evaluation_data(str(dataset_path))

    assert isinstance(data, list)
    assert len(data) > 0
    assert "question" in data[0]
    assert "ground_truth" in data[0]
