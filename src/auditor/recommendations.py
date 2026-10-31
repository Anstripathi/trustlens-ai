"""Map observed findings to specific review actions without promising a fix."""
from __future__ import annotations

from collections import defaultdict

from .schemas import AuditFinding, AuditRecommendation, FindingSeverity

_ACTIONS = (
    ("dataset.missing", "Review missing-data handling",
     "Inspect whether missingness is expected, how it is handled, and whether it differs across meaningful groups."),
    ("dataset.duplicates", "Review repeated records",
     "Confirm whether repeated rows represent valid observations; if deduplicating, do so before splitting data."),
    ("dataset.constant_columns", "Review empty or constant columns",
     "Confirm the column is not needed for downstream controls, then consider excluding it from modeling."),
    ("dataset.identifiers", "Review identifier-like features",
     "Check whether the field identifies records or entities and whether it should be excluded or grouped during splitting."),
    ("dataset.high_cardinality", "Review categorical encoding",
     "Inspect cardinality and rare values; choose an encoding or grouping strategy compatible with the current preprocessing limit."),
    ("dataset.class_imbalance", "Review class imbalance and thresholds",
     "Compare class-wise precision/recall and consider decision-threshold or sampling analysis on validation data."),
    ("dataset.target_leakage", "Investigate potential target leakage",
     "Trace feature generation and split boundaries; the reported association is a warning, not proof of leakage."),
    ("dataset.outliers", "Review flagged distribution tails",
     "Check source and domain context before changing records; IQR flags do not establish data errors."),
    ("model.performance", "Review held-out model performance",
     "Compare class-wise metrics or regression errors with task requirements and a suitable validation baseline."),
    ("model.generalization", "Investigate the fit-to-test gap",
     "Review data splitting, model complexity, and expected deployment data; do not select settings based on the test split."),
    ("reliability.calibration", "Review probability calibration",
     "Inspect calibration bins and sample sizes; fit alternatives on calibration data and keep test data for final reporting."),
    ("reliability.conformal", "Review prediction-interval coverage",
     "Check calibration sample size, exchangeability assumptions, and coverage on a separate representative evaluation set."),
    ("reliability.ood", "Review unusual-input handling",
     "Inspect flagged inputs and define a human review or abstention policy; an OOD flag is not an error verdict."),
    ("reliability.drift", "Investigate distribution changes",
     "Verify schema and input-population changes, then evaluate performance with labels before concluding model degradation."),
    ("security.prompt_injection", "Review prompt boundaries and input handling",
     "Treat scanned instructions as untrusted, enforce instruction hierarchy, and review tool permissions and output controls."),
    ("security.pii", "Protect detected sensitive data",
     "Verify whether the detected value is sensitive; minimize collection, mask it, and restrict access where appropriate."),
    ("security.unsafe_output", "Review unsafe-output handling",
     "Have a qualified reviewer assess context and apply the relevant safety process; lexical matches can be wrong."),
)

_SEVERITY_ORDER = {
    FindingSeverity.CRITICAL: 0,
    FindingSeverity.HIGH: 1,
    FindingSeverity.MEDIUM: 2,
    FindingSeverity.LOW: 3,
    FindingSeverity.INFORMATIONAL: 4,
}


def recommend(findings: tuple[AuditFinding, ...] | list[AuditFinding]) -> tuple[AuditRecommendation, ...]:
    grouped: dict[str, dict] = defaultdict(dict)
    for finding in findings:
        action = next((item for prefix, *item in _ACTIONS if finding.source_check.startswith(prefix)), None)
        if action is None:
            continue
        title, default_guidance = action
        guidance = finding.remediation or default_guidance
        key = finding.source_check
        current = grouped[key]
        current.setdefault("title", title)
        current.setdefault("guidance", guidance)
        current.setdefault("rationale", finding.explanation)
        current.setdefault("priority", finding.severity)
        if _SEVERITY_ORDER[finding.severity] < _SEVERITY_ORDER[current["priority"]]:
            current["priority"] = finding.severity
        current.setdefault("finding_ids", set()).add(finding.finding_id)

    results = [AuditRecommendation(
        recommendation_id=f"recommendation.{key}",
        title=value["title"],
        priority=value["priority"],
        rationale=value["rationale"],
        guidance=value["guidance"],
        finding_ids=tuple(sorted(value["finding_ids"])),
    ) for key, value in grouped.items()]
    return tuple(sorted(results, key=lambda item: (_SEVERITY_ORDER[item.priority], item.title)))
