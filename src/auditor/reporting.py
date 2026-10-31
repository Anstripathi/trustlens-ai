"""Privacy-filtered, in-memory JSON and CSV audit exports."""
from __future__ import annotations

import csv
import io
import json
import math
from datetime import date, datetime
from enum import Enum
from typing import Any

from ..security.schemas import redact_text
from .schemas import AuditResult

REPORT_LIMITATIONS = (
    "This report is an evidence-backed screening summary, not a safety, fairness, trustworthiness, or production-readiness certification.",
    "No raw dataset rows, prompts, model outputs, or unmasked PII are intentionally included.",
    "Heuristic findings can be false positives or miss issues; human review remains necessary.",
)


def _safe(value: Any):
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "item"):
        try:
            return _safe(value.item())
        except (ValueError, TypeError):
            pass
    if isinstance(value, dict):
        return {redact_text(str(key)): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_safe(item) for item in value]
    if isinstance(value, str):
        return redact_text(value)
    if value is None or isinstance(value, (bool, int)):
        return value
    return redact_text(str(value))


def build_report(result: AuditResult) -> dict:
    """Serialize allowlisted assessment fields; input objects are not traversed."""
    checks = []
    for check in result.checks:
        checks.append({
            "check_id": check.check_id,
            "category": check.category.value,
            "status": check.status.value,
            "summary": check.summary,
            "metrics": check.metrics,
            "limitations": list(check.limitations),
            "finding_ids": [finding.finding_id for finding in check.findings],
        })
    findings = []
    for check in result.checks:
        for finding in check.findings:
            findings.append({
                "finding_id": finding.finding_id,
                "category": finding.category.value,
                "severity": finding.severity.value,
                "title": finding.title,
                "explanation": finding.explanation,
                "evidence": finding.evidence,
                "source_check": finding.source_check,
                "heuristic": finding.heuristic,
                "remediation": finding.remediation,
                "human_review_status": "Not reviewed",
            })
    risk = result.risk
    report = {
        "assessment_id": result.assessment_id,
        "created_at": result.created_at,
        "dataset_metadata": result.dataset_metadata,
        "model_metadata": result.model_metadata,
        "checks": checks,
        "checks_executed": sum(check["status"] in ("passed", "finding") for check in checks),
        "checks_passed": sum(check["status"] == "passed" for check in checks),
        "findings": findings,
        "severity_distribution": (risk.severity_distribution if risk is not None else {}),
        "risk": None if risk is None else {
            "risk_level": risk.risk_level,
            "score": risk.score,
            "category_scores": risk.category_scores,
            "category_weights": risk.category_weights,
            "evaluated_categories": risk.evaluated_categories,
            "evaluated_checks": risk.evaluated_checks,
            "unique_findings": risk.unique_findings,
            "duplicate_findings_removed": risk.duplicate_findings_removed,
            "methodology": risk.methodology,
            "limitations": risk.limitations,
        },
        "recommendations": [{
            "recommendation_id": rec.recommendation_id,
            "title": rec.title,
            "priority": rec.priority.value,
            "rationale": rec.rationale,
            "guidance": rec.guidance,
            "finding_ids": rec.finding_ids,
        } for rec in result.recommendations],
        "failed_stages": list(result.failed_stages),
        "failed_checks": [check["check_id"] for check in checks if check["status"] == "failed"],
        "skipped_checks": [check["check_id"] for check in checks
                           if check["status"] in ("not_applicable", "not_evaluated")],
        "limitations": REPORT_LIMITATIONS,
        "human_review_required": True,
    }
    severity_counts: dict[str, int] = {}
    for finding in findings:
        severity_counts[finding["severity"]] = severity_counts.get(finding["severity"], 0) + 1
    report["severity_distribution"] = severity_counts
    return _safe(report)


def json_report_bytes(result: AuditResult) -> bytes:
    return json.dumps(build_report(result), indent=2, ensure_ascii=False,
                      allow_nan=False, sort_keys=True).encode("utf-8")


def _spreadsheet_safe(value: Any) -> str:
    text = "" if value is None else str(value)
    if text.lstrip().startswith(("=", "+", "-", "@", "\t", "\r")):
        return "'" + text
    return text


def csv_report_bytes(result: AuditResult) -> bytes:
    report = build_report(result)
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(["record_type", "assessment_id", "category", "check_id", "status", "severity",
                     "title", "summary", "evidence_or_metrics_json", "recommendation", "human_review_status"])
    for check in report["checks"]:
        writer.writerow([_spreadsheet_safe("check"), _spreadsheet_safe(report["assessment_id"]),
                         _spreadsheet_safe(check["category"]), _spreadsheet_safe(check["check_id"]),
                         _spreadsheet_safe(check["status"]), "", "", _spreadsheet_safe(check["summary"]),
                         _spreadsheet_safe(json.dumps({"metrics": check["metrics"],
                                                       "limitations": check["limitations"]}, sort_keys=True)),
                         "", ""])
    for finding in report["findings"]:
        writer.writerow([_spreadsheet_safe("finding"), _spreadsheet_safe(report["assessment_id"]),
                         _spreadsheet_safe(finding["category"]), _spreadsheet_safe(finding["source_check"]),
                         _spreadsheet_safe("finding"), _spreadsheet_safe(finding["severity"]),
                         _spreadsheet_safe(finding["title"]), _spreadsheet_safe(finding["explanation"]),
                         _spreadsheet_safe(json.dumps(finding["evidence"], sort_keys=True)),
                         _spreadsheet_safe(finding["remediation"]), _spreadsheet_safe(finding["human_review_status"])])
    for rec in report["recommendations"]:
        writer.writerow([_spreadsheet_safe("recommendation"), _spreadsheet_safe(report["assessment_id"]),
                         "", _spreadsheet_safe(rec["recommendation_id"]), "",
                         _spreadsheet_safe(rec["priority"]), _spreadsheet_safe(rec["title"]),
                         _spreadsheet_safe(rec["rationale"]), _spreadsheet_safe(json.dumps(rec["finding_ids"])),
                         _spreadsheet_safe(rec["guidance"]), ""])
    return output.getvalue().encode("utf-8")
