"""Transparent ordinal risk prioritization; the score is not attack probability."""
from __future__ import annotations

from collections import Counter, defaultdict

from .schemas import Finding, Severity

SEVERITY_POINTS = {
    Severity.INFORMATIONAL: 0,
    Severity.LOW: 10,
    Severity.MEDIUM: 25,
    Severity.HIGH: 50,
    Severity.CRITICAL: 75,
}
STRENGTH_MULTIPLIERS = {"High": 1.0, "Medium": 0.75, "Low": 0.5}
CATEGORY_MULTIPLIERS = {
    "email": 1.0, "phone": 1.0, "government_id": 1.15, "payment_card": 1.2,
    "api_key": 1.2, "password": 1.2, "private_key": 1.2,
    "ip_address": 0.9, "prompt_injection": 1.1,
}
RISK_THRESHOLDS = ((15, "Low Risk"), (35, "Moderate Risk"), (60, "High Risk"), (101, "Critical Risk"))
RISK_METHODOLOGY = (
    "Sum per-finding severity points (Informational 0, Low 10, Medium 25, High 50, Critical 75), "
    "multiply by the documented detector-strength and category factors, then cap the total at 100. "
    "Severity points, strength factors (High 1.0, Medium 0.75, Low 0.5), category factors, and tier boundaries "
    "are prioritization choices, not probabilities. Multiple findings add; the score does not certify security."
)


def score_findings(findings: list[Finding], *, completed_checks: int) -> dict:
    """Compute a repeatable 0–100 prioritization score from completed checks only."""
    if isinstance(completed_checks, bool) or not isinstance(completed_checks, int) or completed_checks < 0:
        raise ValueError("completed_checks must be a non-negative integer.")
    if completed_checks == 0:
        return {"available": False, "score": None, "risk_level": "Not assessed",
                "finding_count": len(findings), "category_points": {}, "severity_counts": {},
                "methodology": RISK_METHODOLOGY}

    category_points: dict[str, float] = defaultdict(float)
    raw_points = 0.0
    for finding in findings:
        severity_points = SEVERITY_POINTS[finding.severity]
        strength = STRENGTH_MULTIPLIERS.get(finding.strength, 0.5)
        category_weight = CATEGORY_MULTIPLIERS.get(finding.category, 1.0)
        evidence_factor = 1.0 if finding.evidence and "omitted" not in finding.evidence.lower() else 0.75
        contribution = severity_points * strength * category_weight * evidence_factor
        category_points[finding.category] += contribution
        raw_points += contribution

    score = round(min(100.0, raw_points), 1)
    risk_level = next(label for boundary, label in RISK_THRESHOLDS if score < boundary)
    counts = Counter(finding.severity.value for finding in findings)
    return {
        "available": True,
        "score": score,
        "risk_level": risk_level,
        "finding_count": len(findings),
        "category_points": {key: round(value, 1) for key, value in sorted(category_points.items())},
        "severity_counts": {severity.value: counts.get(severity.value, 0) for severity in Severity},
        "methodology": RISK_METHODOLOGY,
        "completed_checks": completed_checks,
        "limitations": "This ordinal score reflects only selected checks that completed; it is not a probability or security verdict.",
    }
