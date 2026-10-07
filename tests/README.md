# Test Structure

The repository separates Hotel AI Assistant product tests from the RAG evaluation lab. The default pytest configuration discovers tests under this directory and excludes tests marked `live`.

```text
tests/
├── api/                    Flask endpoint behavior
├── unit/                   Hotel Assistant backend/service unit tests
└── rag_evaluation/
    ├── data/               generation and retrieval golden datasets
    └── frameworks/         Custom, RAGAS, and DeepEval adapters and suites
```

## Test areas

| Area | Purpose |
|---|---|
| `api/` | Verifies request validation, processing/status endpoints, and operator-call behavior. |
| `unit/` | Tests the Hotel AI Assistant's backend managers and LLM-service contracts in isolation. |
| `rag_evaluation/` | Evaluates generation and retrieval behavior against golden datasets with Custom, RAGAS, and DeepEval. |

`tests/unit/` contains unit tests for Hotel Assistant backend services. Reporting and dashboard tests are located under `reporting/tests/` and `reporting/dashboard/tests/`.

Each RAG evaluation suite loads test cases from `tests/rag_evaluation/data/`. Shared pytest fixtures provide configured services, stable case identities, Assistant-response reuse, and result recording.

## Running tests

Run the default non-live suite:

```bash
pytest
```

Run only application unit or API tests:

```bash
pytest tests/unit
pytest tests/api
```

Live model-dependent evaluations are opt-in and can be selected by framework or suite. See the [AI Test Lab README](rag_evaluation/README.md) for the evaluation architecture, coverage, thresholds, and commands.

Evaluation output is described in the [reporting and dashboard README](../reporting/README.md).
