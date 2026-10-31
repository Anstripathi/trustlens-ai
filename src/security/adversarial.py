"""Bounded, local assessment runner with scanner failure isolation."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import uuid

import pandas as pd

from . import jailbreak as jailbreak_engine
from . import pii_scanner, prompt_injection, toxicity
from .risk_scoring import score_findings
from .schemas import MAX_DATASET_CELLS, MAX_SECURITY_INPUT_CHARS, Finding, ScanError, validate_text

CHECK_NAMES = ("prompt_injection", "jailbreak", "pii", "toxicity")
MAX_DATASET_CHARS = 100_000


def _source_findings(items: list[Finding], source: str) -> list[Finding]:
    return [replace(item, source=source) for item in items]


def _run_check(checks: dict, findings: list[Finding], name: str, action) -> None:
    try:
        output = action()
        if isinstance(output, tuple):
            new_findings, details = output
        else:
            new_findings, details = output, {}
        findings.extend(new_findings)
        checks[name] = {"status": "completed", "finding_count": len(new_findings), **details}
    except Exception:  # noqa: BLE001 - do not expose scanner exceptions or input details
        checks[name] = {"status": "failed", "finding_count": 0,
                        "message": "This check failed unexpectedly; other selected checks continued."}


def run_assessment(*, prompt: str = "", model_output: str = "",
                   selected_checks: list[str] | tuple[str, ...] | None = None,
                   jailbreak_categories: list[str] | tuple[str, ...] | None = None,
                   pii_categories: list[str] | tuple[str, ...] | None = None,
                   pii_custom_labels: list[str] | tuple[str, ...] = (),
                   toxicity_categories: list[str] | tuple[str, ...] | None = None,
                   dataset: pd.DataFrame | None = None,
                   dataset_columns: list[str] | tuple[str, ...] = (),
                   dataset_max_cells: int = MAX_DATASET_CELLS) -> dict:
    """Assess supplied text and optional current-session data without persisting raw input."""
    prompt = validate_text(prompt)
    model_output = validate_text(model_output)
    if isinstance(dataset_max_cells, bool) or not isinstance(dataset_max_cells, int) or not 1 <= dataset_max_cells <= MAX_DATASET_CELLS:
        raise ScanError(f"Dataset cell limit must be between 1 and {MAX_DATASET_CELLS:,}.")
    selected = set(CHECK_NAMES if selected_checks is None else selected_checks)
    if selected - set(CHECK_NAMES):
        raise ScanError("Unknown security check selected.")
    columns = list(dataset_columns)
    if dataset is None and columns:
        raise ScanError("A dataset must be provided when dataset columns are selected.")
    if dataset is not None:
        if not isinstance(dataset, pd.DataFrame):
            raise ScanError("Dataset scanning requires a validated tabular dataset.")
        unknown_columns = set(columns) - set(dataset.columns)
        if unknown_columns:
            raise ScanError("A selected dataset column is unavailable.")

    checks = {name: {"status": "skipped", "finding_count": 0,
                     "message": "Not selected." if name not in selected else "Required input was not supplied."}
              for name in CHECK_NAMES}
    findings: list[Finding] = []

    if "prompt_injection" in selected:
        if prompt.strip():
            _run_check(checks, findings, "prompt_injection",
                       lambda: _source_findings(prompt_injection.scan_prompt_injection(prompt), "Prompt"))
        else:
            checks["prompt_injection"]["message"] = "Enter a prompt to run this check."

    if "jailbreak" in selected:
        def run_jailbreak():
            cases = jailbreak_engine.run_jailbreak_suite(jailbreak_categories)
            case_results = []
            for case in cases:
                case_results.append({
                    "case_id": case["case_id"],
                    "category": case["category"],
                    "test_input": case["prompt"],
                    "flagged": case["flagged"],
                    "rule_ids": case["rule_ids"],
                    "reason": case["reason"],
                })
            return [], {"case_count": len(cases),
                                   "flagged_case_count": sum(case["flagged"] for case in cases),
                                   "case_results": case_results}
        _run_check(checks, findings, "jailbreak", run_jailbreak)

    if "pii" in selected:
        has_dataset = dataset is not None and bool(columns)
        if prompt.strip() or model_output.strip() or has_dataset:
            def run_pii():
                pii_findings: list[Finding] = []
                if prompt:
                    pii_findings.extend(_source_findings(
                        pii_scanner.scan_pii(prompt, enabled_categories=pii_categories,
                                             custom_labels=pii_custom_labels), "Prompt"))
                if model_output:
                    pii_findings.extend(_source_findings(
                        pii_scanner.scan_pii(model_output, enabled_categories=pii_categories,
                                             custom_labels=pii_custom_labels), "Model output"))
                cells_scanned, cells_examined, skipped_cells, chars_scanned = 0, 0, 0, 0
                partial = False
                if has_dataset:
                    assert dataset is not None
                    total_selected_cells = len(dataset) * len(columns)
                    for column in columns:
                        for row_number, (_, value) in enumerate(dataset[column].items(), start=1):
                            if cells_examined >= dataset_max_cells:
                                partial = cells_examined < total_selected_cells
                                break
                            if chars_scanned >= MAX_DATASET_CHARS:
                                partial = True
                                break
                            cells_examined += 1
                            if pd.isna(value):
                                continue
                            cell_text = str(value)
                            if len(cell_text) > MAX_SECURITY_INPUT_CHARS or chars_scanned + len(cell_text) > MAX_DATASET_CHARS:
                                skipped_cells += 1
                                partial = True
                                continue
                            source = f"Dataset column {columns.index(column) + 1}, cell {row_number}"
                            pii_findings.extend(_source_findings(
                                pii_scanner.scan_pii(cell_text, enabled_categories=pii_categories,
                                                     custom_labels=pii_custom_labels), source))
                            chars_scanned += len(cell_text)
                            cells_scanned += 1
                        if partial and (cells_examined >= dataset_max_cells or chars_scanned >= MAX_DATASET_CHARS):
                            break
                details = {"dataset_cells_examined": cells_examined,
                           "dataset_cells_scanned": cells_scanned,
                           "dataset_cells_skipped": skipped_cells}
                if partial:
                    details.update(status="partial", message="Dataset scan reached a resource limit; some selected cells were not checked.")
                return pii_findings, details
            _run_check(checks, findings, "pii", run_pii)
            if checks["pii"].get("dataset_cells_skipped", 0):
                checks["pii"]["status"] = "partial"
        else:
            checks["pii"]["message"] = "Enter text or select dataset columns to scan."

    if "toxicity" in selected:
        if model_output.strip():
            _run_check(checks, findings, "toxicity",
                       lambda: _source_findings(
                           toxicity.scan_unsafe_output(model_output, enabled_categories=toxicity_categories),
                           "Model output"))
        else:
            checks["toxicity"]["message"] = "Enter a model output to run this check."

    available_checks = sum(check["status"] in ("completed", "partial") for check in checks.values())
    risk = score_findings(findings, completed_checks=available_checks)
    selected_incomplete = any(checks[name]["status"] not in ("completed", "partial") for name in selected)
    assessment_status = "partial" if available_checks and selected_incomplete else (
        "complete" if available_checks else "not_assessed"
    )
    return {
        "assessment_id": str(uuid.uuid4()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": assessment_status,
        "checks": checks,
        "findings": findings,
        "risk": risk,
        "input_sources": [name for name, present in (
            ("prompt", bool(prompt)), ("model_output", bool(model_output)),
            ("dataset", dataset is not None and bool(columns))) if present],
        "limitations": "Rule-based and heuristic checks only. Scanned prompts and outputs are not sent externally."
    }
