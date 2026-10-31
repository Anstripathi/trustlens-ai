"""Split-conformal regression intervals and an Isolation-Forest out-of-distribution flag."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from . import config

MAX_OOD_FIT_ROWS = 10_000
MAX_OOD_CAL_ROWS = 10_000


def conformal_quantile(cal_residuals, alpha: float) -> float | None:
    r = np.sort(np.abs(np.asarray(cal_residuals, float)))
    n = len(r)
    k = math.ceil((n + 1) * (1 - alpha))
    return None if k > n or n == 0 else float(r[k - 1])


def conformal_intervals(result, X: pd.DataFrame | None = None, alpha: float = config.DEFAULT_ALPHA) -> dict:
    """Split conformal prediction: interval = prediction +/- q, q from calibration-split residuals."""
    if result.task != "regression":
        return {"available": False, "reason": "Intervals are implemented for regression only."}
    if not 0 < alpha < 1:
        return {"available": False, "reason": "alpha must be between 0 and 1."}
    res = np.abs(np.asarray(result.y_cal, float) - result.pipeline.predict(result.X_cal))
    q = conformal_quantile(res, alpha)
    if q is None:
        return {"available": False,
                "reason": f"{len(res)} calibration rows are too few for a {1 - alpha:.0%} interval; no interval is shown."}
    X = result.X_test if X is None else X
    pred = result.pipeline.predict(X)
    out = pd.DataFrame({"prediction": pred, "lower": pred - q, "upper": pred + q}, index=X.index)
    yt = np.asarray(result.y_test, float)
    pt = result.pipeline.predict(result.X_test)
    cov = float(np.mean(np.abs(yt - pt) <= q))
    return {"available": True, "alpha": alpha, "half_width": q, "intervals": out,
            "empirical_test_coverage": cov, "n_cal": len(res),
            "notes": ["Marginal (average) coverage guarantee assuming new data is exchangeable with calibration data.",
                      "Interval width is constant: it does not adapt to difficult regions of the input space.",
                      "Under drift or on out-of-distribution inputs the guarantee does not hold.",
                      "Empirical test coverage is a noisy finite-sample check."]}


class OODDetector:
    """Isolation Forest on preprocessed features; threshold = quantile of *calibration-split* scores."""

    def __init__(self, flag_rate: float = config.DEFAULT_OOD_FLAG_RATE, seed: int = 0):
        if not 0 < flag_rate < 0.5:
            raise ValueError("flag_rate must be in (0, 0.5)")
        self.flag_rate, self.seed = flag_rate, seed

    def fit(self, result):
        prep = result.pipeline.named_steps["prep"]
        X_fit = result.X_fit
        X_cal = result.X_cal
        if len(X_fit) > MAX_OOD_FIT_ROWS:
            X_fit = X_fit.sample(MAX_OOD_FIT_ROWS, random_state=self.seed)
        if len(X_cal) > MAX_OOD_CAL_ROWS:
            X_cal = X_cal.sample(MAX_OOD_CAL_ROWS, random_state=self.seed)
        Z_fit = prep.transform(X_fit)
        self.prep_ = prep
        self.forest_ = IsolationForest(n_estimators=100, random_state=self.seed).fit(Z_fit)
        s_cal = self.forest_.score_samples(prep.transform(X_cal))
        self.threshold_ = float(np.quantile(s_cal, self.flag_rate))
        self.n_fit_ = len(X_fit)
        self.n_cal_ = len(X_cal)
        return self

    def score(self, X: pd.DataFrame) -> pd.DataFrame:
        s = self.forest_.score_samples(self.prep_.transform(X))
        return pd.DataFrame({"ood_score": s, "is_out_of_distribution": s < self.threshold_}, index=X.index)

    notes = ("An out-of-distribution flag means the input is unusual compared with training data; "
             "it does not prove the prediction is wrong, and an unflagged input is not guaranteed reliable. "
             "Fit and threshold calibration use deterministic samples of at most 10,000 rows each.")


def fit_ood(result, flag_rate: float = config.DEFAULT_OOD_FLAG_RATE) -> OODDetector:
    if len(result.X_cal) < 20:
        raise ValueError("Too few calibration rows to set an OOD threshold (need >= 20).")
    return OODDetector(flag_rate, seed=result.config.seed).fit(result)
