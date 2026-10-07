# AI Quality Engineering / AI Test Lab

This directory is an AI Quality Engineering lab for the [Hotel AI Assistant](../../README.md). It evaluates the system across LLM generation quality, RAG retrieval quality, deterministic constraints, and agreement or disagreement between independent evaluation approaches.

The lab uses three frameworks together:

- **Custom** combines deterministic application-specific guardrails, an LLM-as-a-Judge rubric, and deterministic information-retrieval metrics.
- **RAGAS** contributes native RAG generation and retrieval metrics plus case-specific rubrics.
- **DeepEval** contributes native generation/retrieval metrics and GEval criteria.

The frameworks provide complementary signals without treating similarly named metrics as semantically identical. Deterministic checks catch contract violations, native metrics evaluate specialized quality dimensions, and framework disagreements identify cases for review.

## Architecture

```text
Golden Dataset
      |
      v
Assistant execution
      |
      v
Custom / RAGAS / DeepEval evaluation
      |
      v
Metric thresholds and quality gates
      |
      v
Recorder -> events.jsonl -> latest.json
      |
      v
Evaluation Dashboard
```

Generation tests use a session-scoped in-memory cache keyed by stable `case_id`. When multiple frameworks run in the same pytest session, the Assistant generates each case response once and every framework evaluates that same response. Judge calls remain framework-specific.

A session-scoped `Recorder` gives all results from one pytest session the same `run_id`, appends each completed evaluation to the shared event stream, and refreshes the aggregated report. Retrieval adapters issue a single search per case where the same retrieved contexts feed both the metric and the recorded evidence.

## Directory layout

```text
rag_evaluation/
├── data/
│   ├── generation/         generation golden datasets
│   └── retrieval/          retrieval golden datasets
└── frameworks/
    ├── custom/             deterministic checks, Custom Judge, IR metrics
    ├── ragas/              RAGAS-native metrics and rubrics
    ├── deepeval/           DeepEval-native metrics and GEval
    ├── conftest.py         shared Assistant cache and Recorder
    └── helpers.py          dataset loading and stable case IDs
```

## Golden dataset design

> **ONE THEMATIC TEST FILE -> ONE CORRESPONDING GOLDEN DATASET**

Each test module loads its corresponding dataset. A stable identity is built as `<golden-file>::<case-name>`, allowing results from different frameworks to be compared by the same case.

Representative generation mappings:

| Test module | Golden dataset |
|---|---|
| `test_llm_faithfulness.py` | `data/generation/llm_faithfulness.json` |
| `test_llm_correctness.py` | `data/generation/llm_correctness.json` |
| `test_llm_hallucination.py` | `data/generation/llm_hallucination.json` |
| `test_llm_relevancy.py` | `data/generation/llm_relevancy.json` |
| `test_llm_negative_constraint.py` | `data/generation/llm_negative_constraint.json` |
| `test_llm_brand_consistency.py` | `data/generation/llm_brand_consistency.json` |

Representative retrieval mappings:

| Test module | Golden dataset |
|---|---|
| `test_top_k_retrieval.py` | `data/retrieval/top_k_retrieval.json` |
| `test_ranking_metrics.py` | `data/retrieval/ranking_metrics.json` |
| `test_query_robustness.py` | `data/retrieval/query_robustness.json` |
| `test_metadata_filtering.py` | `data/retrieval/metadata_filtering.json` |
| `test_distance_stratification.py` | `data/retrieval/distance_stratification.json` |

### Framework-specific configuration

A case can define independent evaluation policies in one block:

```json
"evaluation": {
  "custom": {"min_score": 0.75},
  "ragas": {"min_score": 0.90},
  "deepeval": {"min_score": 0.85}
}
```

Suites with more than one metric use a metric-to-threshold map. The correctness dataset, for example, independently gates RAGAS `factual_correctness` and `answer_correctness`, and DeepEval `geval_factual_correctness` and `geval_answer_correctness`:

```json
"ragas": {
  "min_score": {
    "factual_correctness": 0.60,
    "answer_correctness": 0.75
  }
}
```

Custom aggregate ranking thresholds are defined at suite level in `ranking_metrics.json` and apply to metrics computed over the complete query set.

## Generation evaluation

| Suite | What it validates |
|---|---|
| Faithfulness | Checks whether the answer is supported by the supplied context and avoids unsupported claims. |
| Correctness | Checks whether the answer reaches the factually correct conclusion when applying hotel facts, policies, exclusions, or comparisons. |
| Hallucination | Detects invented details, values, or claims that are absent from or conflict with the context. |
| Relevancy | Detects off-topic, incomplete, or partially answered responses, including missed parts of multi-intent questions. |
| Negative Constraint | Checks whether the Assistant recognizes when the context cannot support an answer, reports low confidence, and avoids guessing. |
| Brand Consistency | Checks whether responses maintain the hotel's identity, professional tone, and hospitality voice without persona breaks or AI disclaimers. |

Custom generation checks validate confidence first and, when configured, required facts or prohibited patterns. A deterministic failure is recorded without calling the Judge. Cases that pass those guardrails proceed to the semantic Judge verdict and score gate.

## Retrieval evaluation

| Suite | Coverage |
|---|---|
| Top-K Retrieval | Checks whether the expected document appears in the retrieved top-K and whether the returned contexts are relevant to the query. |
| Ranking Metrics | Checks whether relevant documents are found, ranked early, retrieved completely, and returned with limited irrelevant noise. |
| Query Robustness | Checks whether retrieval still returns the expected result for clean, noisy, misspelled, and paraphrased queries. |
| Metadata Filtering | Checks whether metadata constraints are respected without losing the semantically relevant result. |
| Distance Stratification | Checks whether vector distances fall within expected similarity bands and align with the application's routing threshold. |

## Framework coverage matrix

### Generation

| Quality suite | Custom | RAGAS | DeepEval |
|---|---|---|---|
| Faithfulness | Validates confidence, required facts, and groundedness with the Custom Judge (`llm_faithfulness`). | Measures answer support from context with `Faithfulness`. | Measures answer support from context with `FaithfulnessMetric`. |
| Correctness | Validates required facts and the correctness of derived conclusions (`llm_correctness`). | Measures factual accuracy with `FactualCorrectness` and similarity to the reference with `AnswerCorrectness`. | Uses GEval Factual Correctness and GEval Answer Correctness as separate checks. |
| Hallucination | Detects prohibited content and uses the Custom Judge to assess unsupported claims (`llm_hallucination`). | Applies `InstanceSpecificRubrics` to case-specific anti-hallucination criteria. | Uses `HallucinationMetric` to check whether the answer agrees with the context. |
| Relevancy | Checks confidence, required facts, and complete coverage of the query (`llm_relevancy`). | Measures how directly the response answers the query with `AnswerRelevancy`. | Measures query-response relevance with `AnswerRelevancyMetric`. |
| Negative Constraint | Checks low-confidence behavior and refusal to guess (`llm_negative_constraint`). | Applies `InstanceSpecificRubrics` to case-specific refusal criteria. | Uses GEval Negative Constraint to evaluate refusal and uncertainty behavior. |
| Brand Consistency | Checks required/prohibited phrasing and hotel persona (`llm_brand_consistency`). | Applies `InstanceSpecificRubrics` to case-specific tone and persona criteria. | Uses GEval Brand Consistency to evaluate identity, tone, and persona. |

### Retrieval

| Quality suite | Custom | RAGAS | DeepEval |
|---|---|---|---|
| Top-K Retrieval | Checks expected-document presence and nearest distance with `HitRate@K` and `max_distance`. | Judges whether retrieved contexts answer the query with `ContextRelevance`. | Judges retrieved-context relevance with `ContextualRelevancyMetric`. |
| Ranking Metrics | Measures hit rate, coverage, noise, first-hit rank, and graded ordering with `HitRate@1`, `HitRate@K`, `Recall@K`, `Precision@K`, `MRR`, and `NDCG@K`. | Measures reference support and coverage with `ContextPrecisionWithReference` and `ContextRecall`. | Measures reference-based ordering and coverage with `ContextualPrecisionMetric` and `ContextualRecallMetric`. |
| Query Robustness | Checks expected top result and distance across query variations (`query_robustness`, `max_distance`). | - | - |
| Metadata Filtering | Checks filter compliance and result distance (`metadata_filtering`, `max_distance`). | - | - |
| Distance Stratification | Checks the global routing threshold and expected distance band (`vector_similarity_threshold`, `distance_stratification`). | - | - |

## Scoring and thresholds

Thresholds are framework- and case-specific. Scores from incompatible metrics are not averaged or used to rank frameworks.

### Custom

The Custom Judge returns both a binary `passed` verdict and an integer score from 0 to 4. The score is normalized to 0-1 as `raw_score / 4` and compared with the case's `evaluation.custom.min_score` (currently `0.75` throughout the generation datasets).

A case must satisfy both the binary Judge verdict and the configured normalized score floor. Confidence mismatches, missing required facts, or prohibited-pattern matches can fail before Judge execution; these guardrail results retain their deterministic PASS/FAIL semantics and diagnostic failure stage.

Custom retrieval preserves each metric's native meaning. Boolean checks record `1.0` or `0.0`; ranking metrics use 0-1 minimum floors; raw distance checks are lower-is-better and retain their native thresholds or distance bands.

### RAGAS

Native RAGAS metrics return their own 0-1 scores and are compared with case-specific `min_score` values. Correctness uses two independently gated metrics: `FactualCorrectness` and `AnswerCorrectness`.

Hallucination, Negative Constraint, and Brand Consistency use `InstanceSpecificRubrics`. Its native 1-5 result is retained in metadata as `raw_rubric_score`, then normalized with `(raw - 1) / 4` before comparison with the configured 0-1 threshold. RAGAS thresholds are configured per case and metric.

### DeepEval

DeepEval uses native metrics and GEval criteria for generation and retrieval:

- `FaithfulnessMetric` and `HallucinationMetric` evaluate whether the answer is supported by and consistent with the context.
- GEval Factual Correctness and GEval Answer Correctness evaluate factual accuracy and overall agreement with the expected answer.
- `AnswerRelevancyMetric` evaluates whether the response directly addresses the query.
- GEval Negative Constraint evaluates refusal and uncertainty behavior when the context cannot support an answer.
- GEval Brand Consistency evaluates hotel identity, tone, and persona.
- `ContextualRelevancyMetric`, `ContextualPrecisionMetric`, and `ContextualRecallMetric` evaluate retrieved-context relevance, ranking quality, and reference coverage.

Native metric and GEval scores use a `0.0-1.0` scale in these suites. Each golden-data case defines its DeepEval `min_score`. The recorded metric score is compared with that threshold: `score >= min_score` produces PASS; otherwise it produces FAIL. Suites with multiple metrics, such as Correctness, configure and evaluate each metric independently.

```json
"deepeval": {
  "min_score": {
    "geval_factual_correctness": 0.80,
    "geval_answer_correctness": 0.75
  }
}
```

## Execution model

- Dataset helpers flatten shared generation context or retrieval corpus data onto each case and assign a stable case ID.
- The Assistant uses the configured `CHAT_MODEL`; the Custom Judge uses `JUDGE_MODEL` through the same OpenAI-compatible endpoint.
- RAGAS uses the configured Judge endpoint. Metrics that need embeddings load `EMBEDDING_MODEL` locally through sentence-transformers.
- DeepEval uses `JUDGE_MODEL` through its native Ollama adapter at `http://localhost:11434`.
- A shared session cache avoids duplicate Assistant calls only within the same combined pytest session.
- Custom generation guardrails run before the Judge, avoiding semantic evaluation when an objective contract already failed.
- RAGAS and DeepEval retrieval adapters exclude composite-reference ranking cases that are structurally ineligible for their reference-based metrics.
- Evaluation events from all selected frameworks share one Recorder and `run_id` within a pytest session.

## Limitations

- Similar metric names do not imply identical prompts, evidence use, or semantics across frameworks.
- Framework disagreement is expected and is treated as a diagnostic signal, not automatically resolved in favor of one evaluator.
- Quality dimensions provide an honest comparison vocabulary, which may be more abstract than thematic suite names.
- There is no cross-framework score average or artificial framework ranking.
- Query Robustness, Metadata Filtering, and Distance Stratification are Custom-only.
- Reference-based RAGAS/DeepEval retrieval metrics omit composite-reference cases; Custom aggregate ranking metrics still use the complete ranking dataset.
- Thresholds are calibrated independently for each framework and, where needed, each case or metric.
- Model-based evaluations are nondeterministic and depend on the configured runtime, model versions, and local resources.

## Running evaluations

`pytest.ini` applies `-m "not live"` by default. Normal test execution includes the unmarked Custom retrieval suites and excludes generation and judge-based evaluation.

### Cheap, non-live evaluation

Run Custom retrieval only:

```bash
pytest tests/rag_evaluation/frameworks/custom/retrieval
```

This does not call the Assistant or an evaluation Judge. It still requires the retrieval environment and ChromaDB embedding runtime to be available.

### Live Custom generation

```bash
pytest tests/rag_evaluation/frameworks/custom/generation/nondeterministic -m live
```

### Live RAGAS evaluation

```bash
pytest tests/rag_evaluation/frameworks/ragas -m live
```

### Live DeepEval evaluation

```bash
pytest tests/rag_evaluation/frameworks/deepeval -m live
```

### Combined framework run

To maximize Assistant-response reuse across all selected frameworks in one pytest session:

```bash
pytest tests/rag_evaluation/frameworks -m "live or not live"
```

Live commands require the configured models and dependencies. They may be slow, nondeterministic, and resource-intensive; prefer the narrowest framework, phase, file, or test node that answers the current quality question.

### Optional Custom HTML reports

The Custom framework can produce self-contained pytest-html artifacts independently of the shared JSON reporting pipeline:

```powershell
# Retrieval only
pytest tests\rag_evaluation\frameworks\custom\retrieval --html=reports\custom_retrieval.html --self-contained-html -o render_collapsed=""

# Live generation only
pytest tests\rag_evaluation\frameworks\custom\generation\nondeterministic -m live --html=reports\custom_generation.html --self-contained-html

# Combined Custom evaluation
pytest tests\rag_evaluation\frameworks\custom -m "live or not live" --html=reports\custom_evaluation.html --self-contained-html -o render_collapsed=""
```

## Reporting integration

Every evaluation adapter records a common result schema with framework, phase, suite, case identity, metric, quality dimension, status, score/threshold, evidence, and model identity where relevant. Results are persisted for the dashboard and disagreement analysis.

See [Reporting and Evaluation Dashboard](../../reporting/README.md) for result-file semantics, aggregation scope, dashboard behavior, and limitations.
