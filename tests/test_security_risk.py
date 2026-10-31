import pytest

from src.security.risk_scoring import score_findings
from src.security.schemas import Finding, Severity


def finding(severity, category="prompt_injection", strength="High", evidence="masked excerpt"):
    return Finding(category, "test.rule", severity, "test rule", strength,
                   "Test finding.", evidence, "Review this input.")


def test_no_completed_checks_has_no_score_and_empty_completed_scan_is_low():
    assert score_findings([], completed_checks=0)["risk_level"] == "Not assessed"
    empty = score_findings([], completed_checks=1)
    assert empty["score"] == 0 and empty["risk_level"] == "Low Risk"


@pytest.mark.parametrize("severity,level", [
    (Severity.LOW, "Low Risk"),
    (Severity.MEDIUM, "Moderate Risk"),
    (Severity.HIGH, "High Risk"),
    (Severity.CRITICAL, "Critical Risk"),
])
def test_severity_boundaries_are_transparent_and_deterministic(severity, level):
    result = score_findings([finding(severity)], completed_checks=1)
    assert result["risk_level"] == level
    assert result == score_findings([finding(severity)], completed_checks=1)
    assert "not probabilities" in result["methodology"]


def test_category_strength_and_evidence_quality_affect_contribution():
    standard = score_findings([finding(Severity.HIGH)], completed_checks=1)["score"]
    pii = score_findings([finding(Severity.HIGH, category="payment_card")], completed_checks=1)["score"]
    weak = score_findings([finding(Severity.HIGH, strength="Low", evidence="Matched text omitted.")], completed_checks=1)["score"]
    assert pii > standard > weak
    assert score_findings([finding(Severity.CRITICAL)] * 3, completed_checks=1)["score"] == 100


def test_completed_check_count_must_be_nonnegative_integer():
    with pytest.raises(ValueError):
        score_findings([], completed_checks=-1)
