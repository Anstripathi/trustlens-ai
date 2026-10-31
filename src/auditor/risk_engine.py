"""Deterministic, documented risk prioritization over evaluated audit evidence."""
from __future__ import annotations

from collections import Counter, defaultdict

from .schemas import (
    AuditCategory,
    AuditCheck,
    AuditFinding,
    CheckStatus,
    FindingSeverity,
    RiskSummary,
)

SEVERITY_POINTS = {
    FindingSeverity.INFORMATIONAL: 0.0,
    FindingSeverity.LOW: 10.0,
    FindingSeverity.MEDIUM: 25.0,
    FindingSeverity.HIGH: 50.0,
    FindingSeverity.CRITICAL: 75.0,
}
CATEGORY_WEIGHTS = {
    AuditCategory.DATASET.value: 0.20,
    AuditCategory.MODEL.value: 0.30,
    AuditCategory.EXPLAINABILITY.value: 0.15,
    AuditCategory.RELIABILITY.value: 0.20,
    AuditCategory.SECURITY.value: 0.15,
}
MIN_EVALUATED_CATEGORIES = 3
MIN_EVALUATED_CHECKS = 6
METHODOLOGY = (
    "Each unique finding contributes its severity points (informational 0, low 10, medium 25, high 50, "
    "critical 75) to its category, capped at 100. Evaluated category scores are combined using the "
    "documented category weights, renormalized over evaluated categories only. A score is withheld as "
    "Insufficient Evidence unless dataset and model checks plus at least three categories and six checks "
    "were evaluated. This ordinal prioritization is not a probability of failure or a safety certification."
)


def _deduplicate(findings: tuple[AuditFinding, ...] | list[AuditFinding]):
    unique: dict[tuple, AuditFinding] = {}
    for item in findings:
        column = item.evidence.get("column")
        identity = (item.category.value, item.source_check, item.title.casefold(),
                    item.severity.value, column)
        unique.setdefault(identity, item)
    return tuple(unique.values())


def score_audit(checks: tuple[AuditCheck, ...] | list[AuditCheck]) -> tuple[RiskSummary, tuple[AuditFinding, ...]]:
    completed = [check for check in checks if check.status in (CheckStatus.PASSED, CheckStatus.FINDING)]
    categories = {check.category.value for check in completed}
    all_findings = tuple(
        finding for check in checks
        if check.status in (CheckStatus.PASSED, CheckStatus.FINDING)
        for finding in check.findings
    )
    unique_findings = _deduplicate(all_findings)
    category_points: dict[str, float] = defaultdict(float)
    for finding in unique_findings:
        category_points[finding.category.value] += SEVERITY_POINTS[finding.severity]
    category_scores = {
        category: min(100.0, round(category_points.get(category, 0.0), 1))
        for category in sorted(categories)
    }

    adequate = (
        AuditCategory.DATASET.value in categories
        and AuditCategory.MODEL.value in categories
        and len(categories) >= MIN_EVALUATED_CATEGORIES
        and len(completed) >= MIN_EVALUATED_CHECKS
    )
    score = None
    risk_level = "Insufficient Evidence"
    if adequate:
        weight_total = sum(CATEGORY_WEIGHTS[category] for category in categories)
        score = round(sum(category_scores[category] * CATEGORY_WEIGHTS[category]
                          for category in categories) / weight_total, 1)
        risk_level = ("Low" if score < 15 else "Moderate" if score < 35
                      else "High" if score < 60 else "Critical")
    counts = Counter(finding.severity.value for finding in unique_findings)
    limitations = (
        "Missing, failed, and not-applicable checks do not count as passes.",
        "Only executed categories contribute; insufficient coverage suppresses the overall numeric score.",
        "Risk levels prioritize review and do not establish fairness, safety, trustworthiness, or production readiness.",
    )
    summary = RiskSummary(
        risk_level=risk_level,
        score=score,
        category_scores=category_scores,
        category_weights={category: CATEGORY_WEIGHTS[category] for category in sorted(categories)},
        evaluated_categories=tuple(sorted(categories)),
        evaluated_checks=len(completed),
        total_checks=len(checks),
        unique_findings=len(unique_findings),
        duplicate_findings_removed=len(all_findings) - len(unique_findings),
        severity_distribution={severity.value: counts.get(severity.value, 0)
                               for severity in FindingSeverity},
        methodology=METHODOLOGY,
        limitations=limitations,
    )
    return summary, unique_findings
