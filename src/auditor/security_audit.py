"""Adapter over Milestone 3's local screening without target-model claims."""
from __future__ import annotations

from ..security.adversarial import CHECK_NAMES, run_assessment
from .schemas import AuditCategory, AuditCheck, AuditFinding, AuditRequest, CheckStatus, FindingSeverity

_SEVERITIES = {severity.value.lower(): severity for severity in FindingSeverity}


def _check(check_id, status, summary, *, findings=(), metrics=None, limitations=()):
    return AuditCheck(check_id, AuditCategory.SECURITY, status, summary, tuple(findings),
                      dict(metrics or {}), tuple(limitations))


def _not_evaluated(reason):
    names = ("prompt_injection", "pii", "unsafe_output", "jailbreak_fixture_suite", "target_interaction")
    return tuple(_check(f"security.{name}", CheckStatus.NOT_EVALUATED, reason) for name in names)


def audit_security(request: AuditRequest) -> tuple[AuditCheck, ...]:
    if not request.run_security:
        return _not_evaluated("Security screening was not selected.")
    selected = tuple(name for name in request.security_checks if name in CHECK_NAMES)
    if len(selected) != len(set(request.security_checks)):
        return (_check("security.configuration", CheckStatus.FAILED,
                       "One or more selected security checks are unsupported."),)
    has_dataset_scan = bool(request.dataset is not None and request.dataset_scan_columns)
    try:
        assessment = run_assessment(
            prompt=request.prompt_sample,
            model_output=request.output_sample,
            selected_checks=selected,
            dataset=request.dataset if has_dataset_scan else None,
            dataset_columns=request.dataset_scan_columns if has_dataset_scan else (),
            dataset_max_cells=min(2_000, int(request.execution_limits.get("security_dataset_cells", 500))),
        )
    except Exception:  # noqa: BLE001 - never echo submitted input in audit reports
        return (_check("security.execution", CheckStatus.FAILED,
                       "The selected local security screens could not be completed."),)

    checks = assessment["checks"]
    findings_by_source: dict[str, list[AuditFinding]] = {name: [] for name in (
        "prompt_injection", "pii", "unsafe_output")}
    for finding in assessment.get("findings", []):
        source = "prompt_injection" if finding.category == "prompt_injection" else (
            "unsafe_output" if finding.category in ("abusive_language", "harassment", "threats",
                                                     "self_harm", "sexual_exploitation", "violence", "hate")
            else "pii")
        severity = _SEVERITIES.get(finding.severity.value.lower(), FindingSeverity.INFORMATIONAL)
        findings_by_source[source].append(AuditFinding(
            finding_id=f"security.{finding.rule_id}.{finding.source}",
            category=AuditCategory.SECURITY,
            severity=severity,
            title=finding.category.replace("_", " ").title(),
            explanation=finding.explanation,
            evidence={"source_excerpt": "Omitted from the consolidated audit report.",
                      "source": finding.source, "detector": finding.detector,
                      "strength": finding.strength},
            source_check=f"security.{source}",
            heuristic=True,
            remediation=finding.remediation,
        ))

    check_map = {
        "prompt_injection": ("prompt_injection", "Prompt injection patterns"),
        "pii": ("pii", "Sensitive-data patterns"),
        "toxicity": ("unsafe_output", "Unsafe-output rules"),
    }
    output = []
    for source_name, (audit_name, label) in check_map.items():
        raw = checks.get(source_name, {})
        status = raw.get("status", "skipped")
        completed = status in ("completed", "partial")
        failed = status == "failed"
        audit_status = (CheckStatus.FAILED if failed else CheckStatus.FINDING
                        if findings_by_source[audit_name] else CheckStatus.PASSED) if completed else CheckStatus.NOT_EVALUATED
        message = raw.get("message")
        if not completed and not failed:
            message = ("No prompt or model output was supplied." if source_name in ("prompt_injection", "toxicity")
                       else "Supply text or explicitly select dataset columns for PII screening.")
        metrics = {"finding_count": len(findings_by_source[audit_name])}
        for key in ("dataset_cells_examined", "dataset_cells_scanned", "dataset_cells_skipped"):
            if key in raw:
                metrics[key] = int(raw[key])
        output.append(_check(
            f"security.{audit_name}", audit_status,
            f"{label} {'completed' if completed else 'failed' if failed else 'not evaluated'}"
            + (f": {message}" if message else "."),
            findings=findings_by_source[audit_name], metrics=metrics,
            limitations=("Local pattern/lexical screening only; false positives and missed detections are possible.",),
        ))

    jailbreak = checks.get("jailbreak", {})
    if "jailbreak" in selected and jailbreak.get("status") in ("completed", "partial"):
        output.append(_check(
            "security.jailbreak_fixture_suite", CheckStatus.NOT_EVALUATED,
            "Local jailbreak fixtures exercised scanner rules; they did not interact with or test a target model.",
            metrics={"fixture_cases": int(jailbreak.get("case_count", 0)),
                     "fixture_matches": int(jailbreak.get("flagged_case_count", 0))},
            limitations=("Fixture detection is not evidence of target-model robustness.",),
        ))
    else:
        output.append(_check("security.jailbreak_fixture_suite", CheckStatus.NOT_EVALUATED,
                             "The local jailbreak fixture suite was not selected."))

    output.append(_check(
        "security.target_interaction", CheckStatus.NOT_EVALUATED,
        "No target LLM interaction is supported by this audit; target-model security remains untested.",
    ))
    return tuple(output)
