# Evaluation Reporting and Dashboard

The reporting subsystem persists [AI Test Lab](../tests/rag_evaluation/README.md) outcomes in a shared schema, aggregates the latest usable state, and exposes results through a read-only local dashboard. The Hotel AI Assistant application is documented in the [root README](../README.md).

<table>
  <tr>
    <td width="50%">
      <img src="https://github.com/user-attachments/assets/5ac14dd7-71e8-4bf7-afbe-e1a9844fb0a5" width="100%" />
    </td>
    <td width="50%">
      <img src="https://github.com/user-attachments/assets/23ab11ca-e8f4-4232-9c61-09e467725d70" width="100%" />
    </td>
  </tr>
</table>

## Architecture

```text
Evaluation adapter
      |
      v
schema.py -> recorder.py -> reports/results/events.jsonl
                               |
                               v
                         aggregate.py
                               |
                               v
                    reports/results/latest.json
                               |
                               v
                dashboard/data.py -> dashboard/app.py
                                      |
                                      v
                               browser dashboard
```

| Component | Responsibility |
|---|---|
| `schema.py` | Defines `EvaluationResult` and PASS, FAIL, BASELINE, N/A, and ERROR statuses. |
| `recorder.py` | Appends one JSON event per completed evaluation and assigns evaluation/run identities. |
| `aggregate.py` | Folds events by stable result key and keeps both the latest attempt and latest valid measurement. |
| `dimensions.py` | Maps framework-native metric names to shared quality dimensions without changing their scores. |
| `paths.py` | Defines the canonical event and aggregate result paths. |
| `dashboard/data.py` | Loads reports, selects scope, builds summaries, and detects disagreements. |
| `dashboard/app.py` | Serves the local Flask API and static dashboard. |
| `dashboard/static/` | Implements the browser UI, filters, tables, and detail dialog. |

Reporting tests are located under `reporting/tests/` and `reporting/dashboard/tests/`.

## Result files

```text
reports/results/events.jsonl
reports/results/latest.json
```

- `events.jsonl` is the append-only evaluation event stream. Each line is a complete evaluation result with a unique `evaluation_id` and a pytest-session `run_id`.
- `latest.json` is the aggregated dashboard source. For each `(framework, phase, case_id, metric)` key, it keeps `latest_attempt` and `latest_valid` pointers.

`latest_attempt` is the most recent record for a result key. `latest_valid` is its most recent PASS, FAIL, or BASELINE measurement and is displayed separately when the latest attempt is ERROR or N/A. Malformed JSONL lines are skipped during aggregation.

## Starting the dashboard

From the project root:

```bash
python -m reporting.dashboard.app
```

Open `http://127.0.0.1:5050/`. The dashboard is independent of the guest application and does not require the Assistant, ChromaDB, or Telegram services merely to view existing results.

## Dashboard behavior

The current UI provides:

- summary cards for total, PASS, FAIL, frameworks, and generation/retrieval counts;
- per-framework status counts and gated pass rates;
- a quality-dimension matrix that keeps native metrics separate;
- framework, phase, quality-dimension, and metric filters;
- case/query text search and a disagreements-only filter;
- a result table with score, threshold, status, and Judge reason;
- row details for case identity, inputs, evidence, models, metadata, and previous valid results;
- normalized score bars and threshold markers when values use a 0-1 scale;
- native RAGAS rubric scores when `raw_rubric_score` is available; and
- a separate Aggregate Retrieval metrics table for Custom suite-level ranking results.

The dashboard API reads `latest.json` first. If it is absent and `events.jsonl` exists, the server aggregates the event stream in memory without writing a replacement file.

## PASS, FAIL, and BASELINE

- **PASS**: a numeric result was produced and met the configured quality gate.
- **FAIL**: a numeric or deterministic result was produced and did not meet the gate.
- **BASELINE**: a numeric result exists, but no threshold was configured for that event, so it has no gate verdict.

N/A represents a completed evaluation without a meaningful measurement, while ERROR represents an evaluation-pipeline failure. Pass rate is calculated as `PASS / (PASS + FAIL)`; BASELINE, N/A, and ERROR are excluded.

## Current and All scopes

The **Current** scope selects the most recent run for each `(framework, phase, suite)` group and shows entries whose latest attempt belongs to that run. A current view can therefore contain results from multiple pytest sessions.

The **All** scope shows every entry retained by `latest.json` without current-run filtering. `latest.json` contains one aggregate entry per result key; `events.jsonl` contains the complete event sequence.

## Framework disagreements

Disagreement matching uses the stable tuple `(phase, case_id, quality_dimension)`. The complete stored `case_id`, containing the golden filename and case name, is used for comparison.

A disagreement is surfaced only when:

1. at least two frameworks evaluated the same comparable identity;
2. each contributed a gated PASS or FAIL verdict; and
3. those verdicts differ.

Aggregate rows are excluded. BASELINE, N/A, and ERROR are not disagreement verdicts and do not create a disagreement.

## Aggregate Retrieval metrics

Custom ranking evaluation records `HitRate@1`, `HitRate@K`, `Recall@K`, `Precision@K`, `MRR`, and `NDCG@K` against one synthetic aggregate case representing the complete ranking dataset. The dashboard identifies these rows by their aggregate query and surfaces them in a dedicated section.

These suite-level values are not combined with framework-native per-case scores. The quality-dimension matrix reports counts and metric names; it never averages incompatible framework scores or produces a framework leaderboard.

## Running reporting tests

The root pytest `testpaths` points to `tests/`, so reporting self-tests are selected explicitly:

```bash
pytest reporting/tests reporting/dashboard/tests
```

These tests use temporary result files and Flask's test client. They do not run live evaluations.

## Limitations and non-goals

The dashboard is a local, read-only engineering tool with:

- no reporting database;
- no authentication or multi-user authorization;
- no cloud persistence;
- no evaluation reruns from the dashboard;
- no token, cost, or runtime analytics;
- no historical comparison/trend UI; and
- no artificial averaging or ranking of evaluation frameworks.

Run evaluations through pytest, then reload the dashboard to inspect the persisted results.
