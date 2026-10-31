"""In-memory, redacted JSON and CSV assessment reports."""
from __future__ import annotations

import csv
import io
import json

from .schemas import redact_text

REPORT_LIMITATIONS = [
    "Checks are local rules and heuristics; they can miss issues and produce false positives.",
    "No target LLM was tested unless the assessment explicitly records an external test (not supported here).",
    "A low risk score does not establish that a model, prompt, output, or dataset is secure.",
    "Government-ID patterns are jurisdiction-limited and are not identity verification.",
]


def _sanitize(value):
    if isinstance(value, dict):
        return {str(key): _sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    if isinstance(value, str):
        return redact_text(value)
    return value


def build_report(assessment: dict) -> dict:
    """Select report-safe fields and recursively redact string values."""
    checks = assessment.get("checks", {})
    risk = assessment.get("risk", {})
    safe_checks = {}
    allowed_check_fields = (
        "status", "finding_count", "message", "case_count", "flagged_case_count",
        "dataset_cells_examined", "dataset_cells_scanned", "dataset_cells_skipped",
    )
    for name, check in checks.items():
        safe_checks[name] = {key: check[key] for key in allowed_check_fields if key in check}
    report = {
        "assessment_id": assessment.get("assessment_id"),
        "created_at": assessment.get("created_at"),
        "executed_categories": [name for name, check in checks.items() if check.get("status") in ("completed", "partial")],
        "checks": safe_checks,
        "overall_risk_level": risk.get("risk_level", "Not assessed"),
        "risk_score": risk.get("score"),
        "risk_methodology": risk.get("methodology"),
        "findings_by_severity": risk.get("severity_counts", {}),
        "findings": [finding.to_dict() for finding in assessment.get("findings", [])],
        "limitations": REPORT_LIMITATIONS,
    }
    return _sanitize(report)


def json_report_bytes(assessment: dict) -> bytes:
    return json.dumps(build_report(assessment), indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8")


def _spreadsheet_safe(value) -> str:
    text = "" if value is None else str(value)
    if text.lstrip().startswith(("=", "+", "-", "@", "\t", "\r")):
        return "'" + text
    return text


def csv_report_bytes(assessment: dict) -> bytes:
    report = build_report(assessment)
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(["assessment_id", "created_at", "overall_risk_level", "risk_score",
                     "source", "category", "severity", "rule_id", "detector", "strength",
                     "human_review_status", "explanation", "evidence", "remediation"])
    findings = report["findings"]
    if not findings:
        writer.writerow([_spreadsheet_safe(report["assessment_id"]),
                         _spreadsheet_safe(report["created_at"]),
                         _spreadsheet_safe(report["overall_risk_level"]),
                         _spreadsheet_safe(report["risk_score"]), "", "", "", "", "", "",
                         "No findings in completed checks.", "", ""])
    for finding in findings:
        writer.writerow([_spreadsheet_safe(report["assessment_id"]),
                         _spreadsheet_safe(report["created_at"]),
                         _spreadsheet_safe(report["overall_risk_level"]),
                         _spreadsheet_safe(report["risk_score"]),
                         *[_spreadsheet_safe(finding.get(key)) for key in (
                         "source", "category", "severity", "rule_id", "detector", "strength", "human_review_status",
                             "explanation", "evidence", "remediation")]])
    return output.getvalue().encode("utf-8")
