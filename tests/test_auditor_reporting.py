import csv
import io
import json
from datetime import datetime, timezone

from src.auditor.reporting import build_report, csv_report_bytes, json_report_bytes
from src.auditor.risk_engine import score_audit
from src.auditor.schemas import (
    AuditCategory, AuditCheck, AuditFinding, AuditRecommendation,
    AuditResult, CheckStatus, FindingSeverity,
)


def sample_result():
    finding = AuditFinding(
        finding_id="security.pii.email", category=AuditCategory.SECURITY,
        severity=FindingSeverity.HIGH, title="Email", explanation="Sensitive value detected.",
        evidence={"masked_excerpt": "[REDACTED EMAIL]", "source": "Model output"},
        source_check="security.pii", remediation="Mask personal data.",
    )
    checks = (
        AuditCheck("dataset.data_types", AuditCategory.DATASET, CheckStatus.NOT_EVALUATED,
                   "No expected schema."),
        AuditCheck("security.pii", AuditCategory.SECURITY, CheckStatus.FINDING,
                   "A pattern was matched.", (finding,)),
    )
    risk, _ = score_audit(checks)
    rec = AuditRecommendation("recommendation.pii", "Protect detected sensitive data",
                              FindingSeverity.HIGH, "Sensitive value detected.",
                              "Minimize and mask it.", (finding.finding_id,))
    return AuditResult("audit-id", datetime(2026, 1, 1, tzinfo=timezone.utc),
                       {"rows": 12, "columns": 3, "fingerprint": "abc"},
                       {"model_name": "Random Forest", "task": "classification"},
                       checks, risk=risk, recommendations=(rec,))


def test_json_report_is_serializable_and_excludes_unpassed_inputs():
    report = json.loads(json_report_bytes(sample_result()))
    assert report["assessment_id"] == "audit-id"
    assert report["risk"]["risk_level"] == "Insufficient Evidence"
    assert report["human_review_required"] is True
    assert report["skipped_checks"] == ["dataset.data_types"]
    assert report["recommendations"][0]["finding_ids"] == ["security.pii.email"]
    rendered = json.dumps(report)
    assert "person@example.com" not in rendered
    assert "private prompt" not in rendered
    assert "password=secret" not in rendered


def test_csv_formula_injection_is_escaped_in_finding_fields():
    result = sample_result()
    original = result.checks[1].findings[0]
    dangerous = AuditFinding(
        finding_id="attack", category=AuditCategory.SECURITY, severity=FindingSeverity.LOW,
        title="=HYPERLINK(\"https://invalid\")", explanation="+cmd", evidence={"note": "@SUM(1,1)"},
        source_check="security.pii",
    )
    checks = (result.checks[0], AuditCheck("security.pii", AuditCategory.SECURITY,
                 CheckStatus.FINDING, "test", (original, dangerous)))
    risk, _ = score_audit(checks)
    result = AuditResult(result.assessment_id, result.created_at, result.dataset_metadata,
                         result.model_metadata, checks, risk=risk)
    rows = list(csv.reader(io.StringIO(csv_report_bytes(result).decode("utf-8"))))
    flattened = "\n".join(",".join(row) for row in rows)
    assert "'=HYPERLINK" in flattened
    assert "'+cmd" in flattened
    assert any(cell.startswith("{") and "@SUM(1,1)" in cell for row in rows for cell in row)
    assert not any(cell.startswith("@SUM") for row in rows for cell in row)
