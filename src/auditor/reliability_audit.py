"""Bounded reliability and uncertainty adapters over stored split data."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from ..calibration import assess_calibration
from ..drift import DISCLAIMER as DRIFT_DISCLAIMER, drift_report
from ..uncertainty import conformal_intervals, fit_ood
from .schemas import AuditCategory, AuditCheck, AuditFinding, AuditRequest, CheckStatus, FindingSeverity

MAX_RELIABILITY_ROWS = 10_000
MAX_DRIFT_ROWS = 10_000


def _check(check_id, status, summary, *, findings=(), metrics=None, limitations=()):
    return AuditCheck(check_id, AuditCategory.RELIABILITY, status, summary, tuple(findings),
                      dict(metrics or {}), tuple(limitations))


def _finding(check_id, suffix, severity, title, explanation, evidence):
    return AuditFinding(f"{check_id}.{suffix}", AuditCategory.RELIABILITY, severity,
                        title, explanation, evidence, source_check=check_id, heuristic=True)


def _bounded_result(result):
    fit_x, fit_y = result.X_fit, result.y_fit
    cal_x, cal_y = result.X_cal, result.y_cal
    test_x, test_y = result.X_test, result.y_test
    seed = result.config.seed
    if len(cal_x) > MAX_RELIABILITY_ROWS:
        cal_x = cal_x.sample(MAX_RELIABILITY_ROWS, random_state=seed)
        cal_y = cal_y.loc[cal_x.index]
    if len(test_x) > MAX_RELIABILITY_ROWS:
        test_x = test_x.sample(MAX_RELIABILITY_ROWS, random_state=seed)
        test_y = test_y.loc[test_x.index]
    if len(fit_x) > MAX_RELIABILITY_ROWS:
        fit_x = fit_x.sample(MAX_RELIABILITY_ROWS, random_state=seed)
        fit_y = fit_y.loc[fit_x.index]
    return replace(result, X_fit=fit_x, y_fit=fit_y, X_cal=cal_x, y_cal=cal_y,
                   X_test=test_x, y_test=test_y)


def _not_evaluated(reason):
    return tuple(_check(f"reliability.{name}", CheckStatus.NOT_EVALUATED, reason)
                 for name in ("confidence", "calibration", "conformal", "ood", "drift"))


def audit_reliability(request: AuditRequest) -> tuple[AuditCheck, ...]:
    result = request.model
    if not request.run_reliability:
        return _not_evaluated("Reliability checks were not selected.")
    if result is None or request.model_fingerprint != request.dataset_fingerprint:
        return _not_evaluated("A model trained on the current dataset is required.")
    bounded = _bounded_result(result)
    checks: list[AuditCheck] = []

    # Confidence and entropy are only classification outputs, not correctness guarantees.
    if result.task == "classification" and hasattr(result.pipeline, "predict_proba"):
        try:
            probabilities = np.asarray(result.pipeline.predict_proba(bounded.X_test), dtype=float)
            confidence = probabilities.max(axis=1)
            entropy = -np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0)), axis=1)
            normalized_entropy = entropy / np.log(probabilities.shape[1]) if probabilities.shape[1] > 1 else np.zeros_like(entropy)
            predictions = result.pipeline.predict(bounded.X_test)
            accuracy = float(np.mean(predictions == np.asarray(bounded.y_test))) if len(predictions) else None
            checks.append(_check(
                "reliability.confidence", CheckStatus.PASSED,
                "Prediction confidence and entropy were summarized on held-out rows.",
                metrics={"rows": int(len(confidence)), "mean_top_probability": float(np.mean(confidence)),
                         "mean_normalized_entropy": float(np.mean(normalized_entropy)),
                         "heldout_accuracy_for_context": accuracy},
                limitations=("Confidence and entropy describe score distributions; they do not guarantee correctness.",),
            ))
        except Exception:  # noqa: BLE001
            checks.append(_check("reliability.confidence", CheckStatus.FAILED,
                                 "Held-out confidence summaries could not be calculated."))
    else:
        checks.append(_check("reliability.confidence", CheckStatus.NOT_APPLICABLE,
                             "Probability confidence and entropy apply to classification models."))

    if result.task == "classification":
        try:
            calibration = assess_calibration(bounded, method="sigmoid")
            if not calibration.get("available"):
                checks.append(_check("reliability.calibration", CheckStatus.NOT_EVALUATED,
                                     "Calibration was unavailable for the current split sizes.",
                                     limitations=(str(calibration.get("reason", "Insufficient calibration evidence.")),)))
            else:
                calibrated = calibration["calibrated"]
                ece = float(calibrated["ece"])
                findings = []
                if ece > 0.10:
                    findings.append(_finding(
                        "reliability.calibration", "high_ece", FindingSeverity.MEDIUM,
                        "Calibration error above review threshold",
                        "Calibrated expected calibration error exceeds 0.10 on held-out rows; inspect bin counts and the intended operating context.",
                        {"ece": ece, "brier": float(calibrated["brier"]),
                         "test_rows": int(len(bounded.y_test))},
                    ))
                checks.append(_check(
                    "reliability.calibration", CheckStatus.FINDING if findings else CheckStatus.PASSED,
                    "Sigmoid calibration was fit on calibration rows and evaluated on the held-out test sample.",
                    findings=findings,
                    metrics={"method": "sigmoid", "calibration_rows": int(len(bounded.y_cal)),
                             "test_rows": int(len(bounded.y_test)), "raw": {
                                 "brier": float(calibration["raw"]["brier"]),
                                 "ece": float(calibration["raw"]["ece"])},
                             "calibrated": {"brier": float(calibrated["brier"]), "ece": ece}},
                    limitations=("ECE is sensitive to binning and test sample size; confidence does not establish correctness.",),
                ))
        except Exception:  # noqa: BLE001
            checks.append(_check("reliability.calibration", CheckStatus.FAILED,
                                 "Calibration metrics could not be calculated."))
    else:
        checks.append(_check("reliability.calibration", CheckStatus.NOT_APPLICABLE,
                             "Probability calibration applies to classification models."))

    if result.task == "regression":
        try:
            intervals = conformal_intervals(bounded, alpha=0.10)
            if not intervals.get("available"):
                checks.append(_check("reliability.conformal", CheckStatus.NOT_EVALUATED,
                                     "Conformal intervals were unavailable for the current calibration split.",
                                     limitations=(str(intervals.get("reason", "Insufficient calibration evidence.")),)))
            else:
                coverage = float(intervals["empirical_test_coverage"])
                findings = []
                if coverage < 0.80:
                    findings.append(_finding(
                        "reliability.conformal", "low_test_coverage", FindingSeverity.MEDIUM,
                        "Observed interval coverage below review threshold",
                        "Empirical held-out coverage is below 80% for nominal 90% intervals; investigate sample variability and data exchangeability.",
                        {"nominal_coverage": 0.90, "empirical_test_coverage": coverage,
                         "test_rows": int(len(bounded.y_test))},
                    ))
                checks.append(_check(
                    "reliability.conformal", CheckStatus.FINDING if findings else CheckStatus.PASSED,
                    "Split-conformal regression intervals were computed using calibration residuals.",
                    findings=findings,
                    metrics={"nominal_coverage": 0.90, "empirical_test_coverage": coverage,
                             "half_width": float(intervals["half_width"]),
                             "calibration_rows": int(intervals["n_cal"]),
                             "test_rows": int(len(bounded.y_test))},
                    limitations=tuple(intervals["notes"]),
                ))
        except Exception:  # noqa: BLE001
            checks.append(_check("reliability.conformal", CheckStatus.FAILED,
                                 "Conformal intervals could not be calculated."))
    else:
        checks.append(_check("reliability.conformal", CheckStatus.NOT_APPLICABLE,
                             "Conformal prediction intervals are implemented for regression."))

    try:
        detector = fit_ood(bounded)
        sample = bounded.X_test
        scored = detector.score(sample)
        flag_rate = float(scored["is_out_of_distribution"].mean()) if len(scored) else 0.0
        checks.append(_check(
            "reliability.ood", CheckStatus.PASSED,
            "An Isolation Forest threshold was fit on fit rows and calibrated on calibration rows, then applied to held-out rows.",
            metrics={"test_rows": int(len(scored)), "flagged_rows": int(scored["is_out_of_distribution"].sum()),
                     "flagged_rate": flag_rate, "fit_rows_sampled": int(detector.n_fit_),
                     "calibration_rows_sampled": int(detector.n_cal_)},
            limitations=("OOD flags indicate unusual inputs, not incorrect predictions; unflagged rows are not guaranteed reliable.",),
        ))
    except Exception:  # noqa: BLE001
        checks.append(_check("reliability.ood", CheckStatus.NOT_EVALUATED,
                             "OOD assessment was unavailable for the current data split."))

    reference, current = request.reference_dataset, request.current_dataset
    if not isinstance(reference, pd.DataFrame) or not isinstance(current, pd.DataFrame):
        checks.append(_check("reliability.drift", CheckStatus.NOT_EVALUATED,
                             "Upload both reference and current datasets to assess drift."))
    else:
        try:
            ref_sample = reference.sample(min(len(reference), MAX_DRIFT_ROWS), random_state=1729)
            cur_sample = current.sample(min(len(current), MAX_DRIFT_ROWS), random_state=1729)
            report = drift_report(ref_sample, cur_sample)
            summary = report["summary"]
            drift_count = int(summary.get("counts", {}).get("drift", 0))
            schema = report["schema"]
            schema_changes = (len(schema["missing_in_current"]) + len(schema["new_in_current"])
                              + len(schema["type_mismatch"]))
            findings = []
            if drift_count:
                findings.append(_finding(
                    "reliability.drift", "distribution_shift", FindingSeverity.MEDIUM,
                    "Distribution changes detected",
                    "Corrected distribution tests and practical effect thresholds identified feature shifts; this does not establish performance degradation.",
                    {"features_with_drift": drift_count, "features_tested": int(summary.get("tested", 0))},
                ))
            if schema_changes:
                findings.append(_finding(
                    "reliability.drift", "schema_change", FindingSeverity.MEDIUM,
                    "Reference/current schema differs",
                    "The reference and current datasets differ in columns or numeric/text types; review compatibility before comparing model inputs.",
                    {"missing_column_count": len(schema["missing_in_current"]),
                     "new_column_count": len(schema["new_in_current"]),
                     "type_mismatch_count": len(schema["type_mismatch"])},
                ))
            drift_status = (CheckStatus.NOT_EVALUATED if int(summary.get("tested", 0)) == 0 and not schema_changes
                            else CheckStatus.FINDING if findings else CheckStatus.PASSED)
            checks.append(_check(
                "reliability.drift", drift_status,
                "Reference and current distributions were compared with schema checks and corrected tests.",
                findings=findings,
                metrics={"reference_rows": int(len(reference)), "current_rows": int(len(current)),
                         "reference_rows_sampled": int(len(ref_sample)),
                         "current_rows_sampled": int(len(cur_sample)),
                         "features_tested": int(summary.get("tested", 0)),
                         "features_with_drift": drift_count,
                         "schema_missing_count": len(schema["missing_in_current"]),
                         "schema_new_count": len(schema["new_in_current"]),
                         "schema_type_mismatch_count": len(schema["type_mismatch"])},
                limitations=(DRIFT_DISCLAIMER,
                             "Only aggregate drift counts are included; feature names and category values are omitted."),
            ))
        except Exception:  # noqa: BLE001
            checks.append(_check("reliability.drift", CheckStatus.FAILED,
                                 "Reference/current drift comparison could not be completed."))
    return tuple(checks)
