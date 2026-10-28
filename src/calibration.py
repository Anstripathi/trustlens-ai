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


def calibration_unavailable_reason(result) -> str | None:
    if result.task != "classification":
        return "Calibration applies to classification only."
    if len(result.classes) != 2:
        return "Calibration is implemented for binary classification only (multiclass shows raw probabilities)."
    y = (np.asarray(result.y_cal) == result.classes[1])
    if len(y) < config.MIN_CAL_ROWS:
        return f"Only {len(y)} calibration rows (need {config.MIN_CAL_ROWS}); calibration would be unreliable."
    if min(y.sum(), (~y).sum()) < config.MIN_CAL_PER_CLASS:
        return f"A class has fewer than {config.MIN_CAL_PER_CLASS} calibration rows."
    return None


def assess_calibration(result, method: str = "sigmoid", n_bins: int = 10) -> dict:
    reason = calibration_unavailable_reason(result)
    if reason:
        return {"available": False, "reason": reason}
    pos = result.classes[1]
    p_cal = result.pipeline.predict_proba(result.X_cal)[:, 1]
    y_cal = (np.asarray(result.y_cal) == pos).astype(int)
    cal = BinaryCalibrator(method).fit(p_cal, y_cal)
    p_raw = result.pipeline.predict_proba(result.X_test)[:, 1]
    p_new = cal.transform(p_raw)
    y_t = (np.asarray(result.y_test) == pos).astype(int)
    return {
        "available": True, "method": method, "positive_class": pos, "calibrator": cal,
        "n_cal": len(y_cal), "n_test": len(y_t),
        "raw": {"brier": brier_score_loss(y_t, p_raw), "ece": expected_calibration_error(y_t, p_raw, n_bins)},
        "calibrated": {"brier": brier_score_loss(y_t, p_new), "ece": expected_calibration_error(y_t, p_new, n_bins)},
        "reliability_raw": reliability_table(y_t, p_raw, n_bins),
        "reliability_calibrated": reliability_table(y_t, p_new, n_bins),
        "notes": ["Calibrator fit on the calibration split only; metrics are on the untouched test split.",
                  "Probabilities describe model scores' agreement with observed frequencies on similar data; "
                  "they are not guarantees of correctness for an individual prediction.",
                  "ECE with few test rows per bin is noisy."],
    }


def confidence_flags(max_probability, threshold: float = config.DEFAULT_CONFIDENCE_THRESHOLD) -> np.ndarray:
    """True where the top-class probability is below the threshold (a low-confidence warning)."""
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0, 1]")
    return np.asarray(max_probability, float) < threshold
