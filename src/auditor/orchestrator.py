"""Small failure-isolating stage runner used by auditor integrations."""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import replace

from .dataset_audit import audit_dataset
from .explainability_audit import audit_explainability
from .model_audit import audit_model
from .recommendations import recommend
from .reliability_audit import audit_reliability
from .risk_engine import score_audit
from .schemas import AuditCategory, AuditCheck, AuditRequest, AuditResult, CheckStatus
from .security_audit import audit_security


AuditStage = Callable[[AuditRequest], Iterable[AuditCheck]]


def run_stages(
    request: AuditRequest,
    stages: Mapping[str, tuple[AuditCategory, AuditStage]],
) -> AuditResult:
    """Run each provided stage independently and retain successful stage results.

    Exception text is intentionally omitted because third-party exceptions can
    contain feature names, row values, prompts or other sensitive input.
    """
    checks: list[AuditCheck] = []
    failed: list[str] = []
    for stage_name, (category, stage) in stages.items():
        try:
            stage_checks = tuple(stage(request))
            if any(not isinstance(check, AuditCheck) for check in stage_checks):
                raise TypeError("stage returned an invalid check result")
            checks.extend(stage_checks)
        except Exception:  # noqa: BLE001 - isolate a component and protect input values
            failed.append(stage_name)
            checks.append(AuditCheck(
                check_id=f"stage.{stage_name}",
                category=category,
                status=CheckStatus.FAILED,
                summary="This assessment stage could not be completed.",
                limitations=("The failure details were withheld to avoid exposing input data.",),
            ))
    return AuditResult.create(checks=tuple(checks), failed_stages=tuple(failed))


def run_audit(request: AuditRequest) -> AuditResult:
    """Run the fixed TrustLens audit stages over the supplied session objects."""
    stages = {
        "dataset": (AuditCategory.DATASET, audit_dataset),
        "model": (AuditCategory.MODEL, audit_model),
        "explainability": (AuditCategory.EXPLAINABILITY, audit_explainability),
        "reliability": (AuditCategory.RELIABILITY, audit_reliability),
        "security": (AuditCategory.SECURITY, audit_security),
    }
    partial = run_stages(request, stages)
    risk, findings = score_audit(partial.checks)
    dataset = request.dataset
    dataset_metadata = ({
        "rows": int(dataset.shape[0]),
        "columns": int(dataset.shape[1]),
        "fingerprint": request.dataset_fingerprint,
    } if dataset is not None and hasattr(dataset, "shape") else {})
    model_metadata = None
    if request.model is not None:
        result = request.model
        config = getattr(result, "config", None)
        split = getattr(result, "split_info", {}) or {}
        model_metadata = {
            "model_name": getattr(config, "model_name", "Unknown"),
            "task": getattr(result, "task", "Unknown"),
            "feature_count": len(getattr(result, "features", ()) or ()),
            "fit_rows": int(split.get("n_fit", len(getattr(result, "X_fit", ())))),
            "validation_rows": int(split.get("n_cal", len(getattr(result, "X_cal", ())))),
            "test_rows": int(split.get("n_test", len(getattr(result, "X_test", ())))),
            "seed": getattr(config, "seed", None),
            "matches_dataset": request.model_fingerprint == request.dataset_fingerprint,
        }
    return replace(
        partial,
        dataset_metadata=dataset_metadata,
        model_metadata=model_metadata,
        risk=risk,
        recommendations=recommend(findings),
    )
