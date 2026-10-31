from src.auditor.recommendations import recommend
from src.auditor.risk_engine import CATEGORY_WEIGHTS, score_audit
from src.auditor.schemas import AuditCategory, AuditCheck, AuditFinding, CheckStatus, FindingSeverity


def check(category, name, status=CheckStatus.PASSED, findings=()):
    return AuditCheck(name, category, status, "test", tuple(findings))


def finding(category=AuditCategory.DATASET, finding_id="missing-a", title="Missing values observed",
            severity=FindingSeverity.MEDIUM, source="dataset.missing", evidence=None):
    return AuditFinding(finding_id, category, severity, title, "Review this evidence.",
                        evidence or {"column": "feature_a"}, source_check=source)


def sufficient_checks(findings=()):
    return [
        check(AuditCategory.DATASET, "dataset.quality", CheckStatus.FINDING if findings else CheckStatus.PASSED, findings),
        check(AuditCategory.MODEL, "model.metrics"),
        check(AuditCategory.EXPLAINABILITY, "explainability.shap"),
        check(AuditCategory.RELIABILITY, "reliability.ood"),
        check(AuditCategory.DATASET, "dataset.profile"),
        check(AuditCategory.MODEL, "model.split"),
    ]


def test_risk_score_is_deterministic_weighted_and_deduplicates_findings():
    item = finding()
    first, unique = score_audit(sufficient_checks((item, item)))
    second, _ = score_audit(sufficient_checks((item, item)))
    assert first.score == second.score
    assert first.risk_level == "Low"
    assert first.duplicate_findings_removed == 1
    assert first.unique_findings == 1
    assert unique == (item,)
    assert sum(first.category_weights.values()) == sum(CATEGORY_WEIGHTS[k] for k in first.evaluated_categories)


def test_insufficient_categories_withhold_score_and_do_not_count_skipped_checks():
    checks = [check(AuditCategory.DATASET, f"data.{n}") for n in range(6)]
    risk, _ = score_audit(checks)
    assert risk.risk_level == "Insufficient Evidence"
    assert risk.score is None
    risk_with_skip, _ = score_audit(checks + [check(AuditCategory.RELIABILITY, "reliability.skip",
                                                     CheckStatus.NOT_EVALUATED)])
    assert risk_with_skip.evaluated_categories == ("dataset",)


def test_failed_check_findings_do_not_affect_risk():
    severe = finding(severity=FindingSeverity.CRITICAL)
    checks = sufficient_checks()
    checks.append(check(AuditCategory.SECURITY, "security.failed", CheckStatus.FAILED, (severe,)))
    risk, unique = score_audit(checks)
    assert not unique
    assert "security" not in risk.evaluated_categories


def test_recommendations_only_follow_matching_findings_and_track_highest_priority():
    low = finding(finding_id="one", severity=FindingSeverity.LOW)
    high = finding(finding_id="two", severity=FindingSeverity.HIGH)
    output = recommend((low, high))
    assert len(output) == 1
    assert output[0].title == "Review missing-data handling"
    assert output[0].priority is FindingSeverity.HIGH
    assert set(output[0].finding_ids) == {"one", "two"}
    assert not recommend(())
