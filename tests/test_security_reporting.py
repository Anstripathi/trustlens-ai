import csv
import io
import json

from src.security.reporting import build_report, csv_report_bytes, json_report_bytes
from src.security.risk_scoring import score_findings
from src.security.schemas import Finding, Severity


def sample_assessment():
    finding = Finding(
        category="email", rule_id="pii.email", severity=Severity.LOW,
        detector="rule", strength="Medium", explanation="Email-like value matched.",
        evidence="Contact [REDACTED EMAIL]", remediation="Remove it.", source="Prompt",
    )
    return {
        "assessment_id": "security-test-1", "created_at": "2026-10-31T00:00:00+00:00",
        "checks": {"pii": {"status": "completed", "finding_count": 1,
                            "prompt": "alice@example.com", "case_results": [{"test_input": "raw prompt"}]},
                   "toxicity": {"status": "failed", "message": "safe failure"},
                   "jailbreak": {"status": "skipped", "message": "Not selected."}},
        "findings": [finding], "risk": score_findings([finding], completed_checks=1),
        "raw_prompt": "alice@example.com", "raw_output": "password=hunter2",
    }


def test_json_report_is_structured_and_excludes_raw_text_and_unmasked_secrets():
    data = json_report_bytes(sample_assessment())
    result = json.loads(data)
    rendered = data.decode()
    assert result["assessment_id"] == "security-test-1"
    assert "failed" == result["checks"]["toxicity"]["status"]
    assert "skipped" == result["checks"]["jailbreak"]["status"]
    assert "alice@example.com" not in rendered
    assert "hunter2" not in rendered and "raw prompt" not in rendered
    assert "Not reviewed" in rendered


def test_csv_is_parseable_formula_safe_and_does_not_include_raw_input():
    assessment = sample_assessment()
    assessment["findings"][0] = Finding(
        category="email", rule_id="pii.email", severity=Severity.LOW,
        detector="rule", strength="Medium", explanation="safe", evidence="masked",
        remediation="review", source="=HYPERLINK(\"https://bad\")",
    )
    raw = csv_report_bytes(assessment).decode("utf-8")
    rows = list(csv.reader(io.StringIO(raw)))
    assert len(rows) == 2 and rows[1][4].startswith("'=")
    assert "alice@example.com" not in raw and "hunter2" not in raw


def test_report_check_fields_are_allowlisted():
    report = build_report(sample_assessment())
    assert "prompt" not in report["checks"]["pii"]
    assert "case_results" not in report["checks"]["pii"]
    assert "raw_prompt" not in report and "raw_output" not in report
