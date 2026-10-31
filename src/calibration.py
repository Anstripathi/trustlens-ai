"""Probability calibration assessment (binary classification), leakage-safe.

The calibrator is fit on the *calibration split* only. Test labels are never used to fit it,
so test-set metrics for raw and calibrated probabilities are both honest hold-out estimates.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from . import config


def reliability_table(y_bin, p, n_bins: int = 10) -> pd.DataFrame:
    y_bin, p = np.asarray(y_bin, float), np.asarray(p, float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            rows.append({"bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}", "mean_predicted": p[m].mean(),
                         "observed_rate": y_bin[m].mean(), "count": int(m.sum())})
    return pd.DataFrame(rows)


def expected_calibration_error(y_bin, p, n_bins: int = 10) -> float:
    t = reliability_table(y_bin, p, n_bins)
    if t.empty:
        return float("nan")
    return float((t["count"] * (t["mean_predicted"] - t["observed_rate"]).abs()).sum() / t["count"].sum())


class BinaryCalibrator:
    def __init__(self, method: str = "sigmoid"):
        if method not in ("sigmoid", "isotonic"):
            raise ValueError("method must be 'sigmoid' or 'isotonic'")
        self.method = method

    @staticmethod
    def _logit(p):
        p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def fit(self, p, y_bin):
        if self.method == "sigmoid":
            self.m_ = LogisticRegression(C=1e6, max_iter=1000).fit(self._logit(p), y_bin)
        else:
            self.m_ = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p, y_bin)
        return self

    def transform(self, p):
        if self.method == "sigmoid":
            return self.m_.predict_proba(self._logit(p))[:, 1]
        return self.m_.predict(np.asarray(p, float))


class MulticlassCalibrator:
    """One-vs-rest class calibrators with row-wise probability renormalization."""

    def __init__(self, method: str, classes):
        if method not in ("sigmoid", "isotonic"):
            raise ValueError("method must be 'sigmoid' or 'isotonic'")
        if len(classes) < 3:
            raise ValueError("Multiclass calibration requires at least three classes.")
        self.method = method
        self.classes = list(classes)

    def fit(self, probabilities, y):
        probabilities = np.asarray(probabilities, dtype=float)
        y = np.asarray(y)
        self.calibrators_ = []
        for index, label in enumerate(self.classes):
            target = (y == label).astype(int)
            self.calibrators_.append(BinaryCalibrator(self.method).fit(probabilities[:, index], target))
        return self

    def transform(self, probabilities):
        probabilities = np.asarray(probabilities, dtype=float)
        calibrated = np.column_stack([
            model.transform(probabilities[:, index])
            for index, model in enumerate(self.calibrators_)
        ])
        totals = calibrated.sum(axis=1, keepdims=True)
        invalid = ~np.isfinite(totals[:, 0]) | (totals[:, 0] <= 0)
        calibrated[~invalid] /= totals[~invalid]
        calibrated[invalid] = 1.0 / len(self.classes)
        return calibrated


def calibration_unavailable_reason(result) -> str | None:
    if result.task != "classification":
        return "Calibration applies to classification only."
    y = (np.asarray(result.y_cal) == result.classes[1])
    if len(y) < config.MIN_CAL_ROWS:
        return f"Only {len(y)} calibration rows (need {config.MIN_CAL_ROWS}); calibration would be unreliable."
    if len(result.classes) == 2:
        if min(y.sum(), (~y).sum()) < config.MIN_CAL_PER_CLASS:
            return f"A class has fewer than {config.MIN_CAL_PER_CLASS} calibration rows."
    else:
        counts = pd.Series(result.y_cal).value_counts()
        if any(counts.get(label, 0) < config.MIN_CAL_PER_CLASS for label in result.classes):
            return f"Every class needs at least {config.MIN_CAL_PER_CLASS} calibration rows."
    return None


def assess_calibration(result, method: str = "sigmoid", n_bins: int = 10) -> dict:
    reason = calibration_unavailable_reason(result)
    if reason:
        return {"available": False, "reason": reason}
    if len(result.classes) == 2:
        pos = result.classes[1]
        p_cal = result.pipeline.predict_proba(result.X_cal)[:, 1]
        y_cal = (np.asarray(result.y_cal) == pos).astype(int)
        cal = BinaryCalibrator(method).fit(p_cal, y_cal)
        p_raw = result.pipeline.predict_proba(result.X_test)[:, 1]
        p_new = cal.transform(p_raw)
        y_t = (np.asarray(result.y_test) == pos).astype(int)
        raw_metrics = {"brier": brier_score_loss(y_t, p_raw),
                       "ece": expected_calibration_error(y_t, p_raw, n_bins)}
        calibrated_metrics = {"brier": brier_score_loss(y_t, p_new),
                              "ece": expected_calibration_error(y_t, p_new, n_bins)}
        reliability_raw = reliability_table(y_t, p_raw, n_bins)
        reliability_calibrated = reliability_table(y_t, p_new, n_bins)
        notes = ["Binary class probability calibration fit on the calibration split; metrics use the untouched test split."]
    else:
        labels = np.asarray(result.classes)
        cal = MulticlassCalibrator(method, labels).fit(
            result.pipeline.predict_proba(result.X_cal), np.asarray(result.y_cal)
        )
        raw_matrix = result.pipeline.predict_proba(result.X_test)
        calibrated_matrix = cal.transform(raw_matrix)
        y_indices = np.array([list(labels).index(value) for value in result.y_test], dtype=int)
        one_hot = np.eye(len(labels))[y_indices]
        raw_confidence = raw_matrix.max(axis=1)
        calibrated_confidence = calibrated_matrix.max(axis=1)
        raw_correct = raw_matrix.argmax(axis=1) == y_indices
        calibrated_correct = calibrated_matrix.argmax(axis=1) == y_indices
        raw_metrics = {"brier": float(np.mean(np.sum((raw_matrix - one_hot) ** 2, axis=1))),
                       "ece": expected_calibration_error(raw_correct.astype(int), raw_confidence, n_bins)}
        calibrated_metrics = {"brier": float(np.mean(np.sum((calibrated_matrix - one_hot) ** 2, axis=1))),
                              "ece": expected_calibration_error(calibrated_correct.astype(int), calibrated_confidence, n_bins)}
        reliability_raw = reliability_table(raw_correct.astype(int), raw_confidence, n_bins)
        reliability_calibrated = reliability_table(calibrated_correct.astype(int), calibrated_confidence, n_bins)
        notes = ["One-vs-rest class calibrators fit on the calibration split and renormalized; top-label reliability and multiclass Brier use the untouched test split."]
    return {
        "available": True, "method": method, "calibrator": cal,
        "n_cal": len(result.y_cal), "n_test": len(result.y_test),
        "raw": raw_metrics,
        "calibrated": calibrated_metrics,
        "reliability_raw": reliability_raw,
        "reliability_calibrated": reliability_calibrated,
        "notes": notes + [
                  "Probabilities describe model scores' agreement with observed frequencies on similar data; "
                  "they are not guarantees of correctness for an individual prediction.",
                  "ECE with few test rows per bin is noisy."],
    }


def confidence_flags(max_probability, threshold: float = config.DEFAULT_CONFIDENCE_THRESHOLD) -> np.ndarray:
    """True where the top-class probability is below the threshold (a low-confidence warning)."""
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0, 1]")
    return np.asarray(max_probability, float) < threshold
