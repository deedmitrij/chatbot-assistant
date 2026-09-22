"""Framework-agnostic result recording and reporting for the AI Test Lab.

Normalizes Custom / RAGAS / (later) DeepEval evaluation results into a
shared EvaluationResult schema, persisted as an append-only JSONL event log
and folded into a materialized "latest per case/metric" view for reporting.
"""
