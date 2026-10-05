"""Local read-only Evaluation Dashboard.

A consumer of the reporting artifacts written by the reporting
package (latest.json, falling back to an in-memory aggregate of events.jsonl). It never
writes to either file and never changes evaluation semantics.
"""
