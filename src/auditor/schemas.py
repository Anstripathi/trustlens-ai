"""Shared contracts for the model auditor.

Dataset/model objects are kept in the request for in-process evaluation only.
Only the result's sanitized metadata, checks, metrics and findings belong in
report serialization; never copy raw rows, prompts or outputs into them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4


class AuditCategory(str, Enum):
    DATASET = "dataset"
    MODEL = "model"
    EXPLAINABILITY = "explainability"
    RELIABILITY = "reliability"
    SECURITY = "security"


class CheckStatus(str, Enum):
    PASSED = "passed"
    FINDING = "finding"
    FAILED = "failed"
    NOT_APPLICABLE = "not_applicable"
    NOT_EVALUATED = "not_evaluated"


class FindingSeverity(str, Enum):
    INFORMATIONAL = "informational"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True)
class AuditFinding:
    """An evidence-backed observation; evidence must be sanitized by its producer."""

    finding_id: str
    category: AuditCategory
    severity: FindingSeverity
    title: str
    explanation: str
    evidence: dict[str, Any] = field(default_factory=dict)
    source_check: str = ""
    heuristic: bool = False
    remediation: str = ""


@dataclass(frozen=True)
class AuditCheck:
    """Outcome of one bounded audit check."""

    check_id: str
    category: AuditCategory
    status: CheckStatus
    summary: str
    findings: tuple[AuditFinding, ...] = ()
    metrics: dict[str, Any] = field(default_factory=dict)
    limitations: tuple[str, ...] = ()


@dataclass(frozen=True)
class AuditRecommendation:
    recommendation_id: str
    title: str
    priority: FindingSeverity
    rationale: str
    guidance: str
    finding_ids: tuple[str, ...]


@dataclass(frozen=True)
class RiskSummary:
    risk_level: str
    score: float | None
    category_scores: dict[str, float | None]
    category_weights: dict[str, float]
    evaluated_categories: tuple[str, ...]
    evaluated_checks: int
    total_checks: int
    unique_findings: int
    duplicate_findings_removed: int
    severity_distribution: dict[str, int]
    methodology: str
    limitations: tuple[str, ...] = ()


@dataclass(frozen=True)
class AuditRequest:
    """In-memory inputs for an audit; raw inputs must never be serialized."""

    dataset: Any | None
    dataset_name: str | None
    dataset_fingerprint: str | None
    model: Any | None = None
    model_fingerprint: str | None = None
    reference_dataset: Any | None = None
    current_dataset: Any | None = None
    target_column: str | None = None
    benchmark_models: bool = False
    run_cross_validation: bool = False
    run_explainability: bool = True
    run_reliability: bool = True
    run_security: bool = False
    security_checks: tuple[str, ...] = ("prompt_injection", "jailbreak", "pii", "toxicity")
    prompt_sample: str = ""
    output_sample: str = ""
    dataset_scan_columns: tuple[str, ...] = ()
    execution_limits: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class AuditResult:
    """Structured assessment envelope. Risk and report fields are added in later phases."""

    assessment_id: str
    created_at: datetime
    dataset_metadata: dict[str, Any]
    model_metadata: dict[str, Any] | None
    checks: tuple[AuditCheck, ...]
    failed_stages: tuple[str, ...] = ()
    risk: RiskSummary | None = None
    recommendations: tuple[AuditRecommendation, ...] = ()

    @classmethod
    def create(
        cls,
        *,
        dataset_metadata: dict[str, Any] | None = None,
        model_metadata: dict[str, Any] | None = None,
        checks: tuple[AuditCheck, ...] = (),
        failed_stages: tuple[str, ...] = (),
        risk: RiskSummary | None = None,
        recommendations: tuple[AuditRecommendation, ...] = (),
    ) -> "AuditResult":
        return cls(
            assessment_id=str(uuid4()),
            created_at=datetime.now(timezone.utc),
            dataset_metadata=dict(dataset_metadata or {}),
            model_metadata=dict(model_metadata) if model_metadata is not None else None,
            checks=checks,
            failed_stages=failed_stages,
            risk=risk,
            recommendations=recommendations,
        )
