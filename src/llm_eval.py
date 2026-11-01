"""Offline, deterministic evaluation helpers for supplied LLM/RAG traces."""
from __future__ import annotations

import re

import pandas as pd

from .security.pii_scanner import scan_pii
from .security.prompt_injection import scan_prompt_injection

MAX_EVAL_ROWS = 2_000
MAX_FIELD_CHARS = 10_000
REQUIRED_FIELDS = {"prompt", "answer", "context"}
_WORD = re.compile(r"[a-z0-9]+", re.I)
_CITATION = re.compile(r"(?:\[[0-9]{1,3}\]|https?://\S+)")


def _tokens(value: str) -> set[str]:
    return set(_WORD.findall(value.casefold()))


def evaluate_traces(frame: pd.DataFrame) -> dict:
    """Summarize trace quality with lexical proxies; no model judge is invoked."""
    if not isinstance(frame, pd.DataFrame) or not REQUIRED_FIELDS.issubset(set(map(str, frame.columns))):
        raise ValueError("Trace table must contain prompt, answer, and context columns.")
    if frame.empty or len(frame) > MAX_EVAL_ROWS:
        raise ValueError(f"Trace table must contain between 1 and {MAX_EVAL_ROWS} rows.")
    columns = {str(c).casefold(): c for c in frame.columns}
    summaries = {"grounding_overlap": [], "citation_present": [], "injection_flags": 0,
                 "pii_findings": 0, "latency_ms": [], "answer_variants": {}}
    for _, row in frame.iterrows():
        prompt, answer, context = (row[columns[key]] for key in ("prompt", "answer", "context"))
        fields = ["" if pd.isna(v) else str(v) for v in (prompt, answer, context)]
        if any(len(v) > MAX_FIELD_CHARS for v in fields):
            raise ValueError("A trace field exceeds the 10,000 character limit.")
        prompt, answer, context = fields
        answer_tokens, context_tokens = _tokens(answer), _tokens(context)
        overlap = len(answer_tokens & context_tokens) / max(1, len(answer_tokens))
        summaries["grounding_overlap"].append(overlap)
        summaries["citation_present"].append(bool(_CITATION.search(answer)))
        summaries["injection_flags"] += int(bool(scan_prompt_injection(prompt)))
        summaries["pii_findings"] += len(scan_pii(answer))
        prompt_key = " ".join(_tokens(prompt))
        if prompt_key:
            summaries["answer_variants"].setdefault(prompt_key, set()).add(" ".join(answer.casefold().split()))
        latency_col = columns.get("latency_ms")
        if latency_col is not None and not pd.isna(row[latency_col]):
            value = float(row[latency_col])
            if value < 0:
                raise ValueError("latency_ms values must be non-negative.")
            summaries["latency_ms"].append(value)
    repeated = [variants for variants in summaries["answer_variants"].values() if len(variants) > 1]
    count = len(frame)
    latency = summaries["latency_ms"]
    return {
        "rows_evaluated": count,
        "mean_lexical_grounding_overlap": sum(summaries["grounding_overlap"]) / count,
        "citation_presence_rate": sum(summaries["citation_present"]) / count,
        "prompt_injection_flag_count": summaries["injection_flags"],
        "output_pii_finding_count": summaries["pii_findings"],
        "repeated_prompt_answer_variance_count": len(repeated),
        "mean_latency_ms": sum(latency) / len(latency) if latency else None,
        "latency_rows": len(latency),
        "limitations": [
            "Lexical overlap and citation presence are deterministic proxies, not semantic groundedness or citation verification.",
            "Repeated-prompt variance detects differing normalized answers only; it is not semantic contradiction detection.",
            "Local pattern scanners are incomplete and do not guarantee prompt-injection or privacy safety.",
            "No LLM judge or external API is called; input traces are not retained in the result.",
        ],
    }
