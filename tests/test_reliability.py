import numpy as np
import pytest
from helpers import clf_df, reg_df
from src.calibration import (BinaryCalibrator, assess_calibration, confidence_flags,
                             expected_calibration_error, reliability_table)
from src.train import TrainConfig, train_model
from src.uncertainty import conformal_intervals, conformal_quantile, fit_ood


def test_reliability_table_and_ece_perfect_and_bad():
    p = np.array([0.1] * 10 + [0.9] * 10); y = np.array([1] + [0] * 9 + [1] * 9 + [0])
    assert expected_calibration_error(y, p) < 1e-9
    assert expected_calibration_error(np.zeros(10), np.full(10, 0.9)) > 0.89
    assert reliability_table(y, p)["count"].sum() == 20


@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
def test_calibrator_fixes_overconfident_scores(method):
    rng = np.random.default_rng(0)
    true_p = rng.uniform(0.2, 0.8, 4000); y = (rng.random(4000) < true_p).astype(int)
    raw = np.clip(0.5 + (true_p - 0.5) * 2.0, 0.01, 0.99)           # overconfident
    cal = BinaryCalibrator(method).fit(raw[:2000], y[:2000])
    assert expected_calibration_error(y[2000:], cal.transform(raw[2000:])) < expected_calibration_error(y[2000:], raw[2000:])


def test_calibration_uses_cal_split_only_and_reports_both():
    r = train_model(clf_df(n=1500), TrainConfig("y", "classification", "Random Forest"))
    a = assess_calibration(r, "sigmoid")
    assert a["available"] and a["n_cal"] == len(r.X_cal) and a["n_test"] == len(r.X_test)
    assert {"brier", "ece"} <= set(a["raw"]) and {"brier", "ece"} <= set(a["calibrated"])


def test_calibration_unavailable_cases():
    small = train_model(clf_df(n=150), TrainConfig("y", "classification", "Decision Tree"))
    assert not assess_calibration(small)["available"]
    reg = train_model(reg_df(), TrainConfig("y", "regression", "Linear Regression"))
    assert not assess_calibration(reg)["available"]
    df = clf_df(n=1000); df["y"] = np.random.default_rng(0).choice(["a", "b", "c"], len(df))
    multi = assess_calibration(train_model(df, TrainConfig("y", "classification", "Decision Tree")))
    assert multi["available"] and multi["n_cal"] > 0
    assert {"brier", "ece"} <= set(multi["calibrated"])
    assert multi["reliability_calibrated"]["count"].sum() > 0
    p = multi["calibrator"].transform(np.array([[0.2, 0.3, 0.5], [0.8, 0.1, 0.1]]))
    assert np.allclose(p.sum(axis=1), 1.0) and np.all((p >= 0) & (p <= 1))


def test_confidence_threshold():
    assert list(confidence_flags([0.5, 0.6, 0.9], 0.6)) == [True, False, False]
    with pytest.raises(ValueError):
        confidence_flags([0.5], 0)


def test_conformal_quantile_math():
    assert conformal_quantile(np.arange(1, 10), 0.2) == 8.0      # k=ceil(10*0.8)=8
    assert conformal_quantile(np.arange(1, 5), 0.1) is None      # k=ceil(5*.9)=5 > 4


def test_conformal_coverage_close_to_nominal_and_sane():
    r = train_model(reg_df(n=3000), TrainConfig("y", "regression", "Random Forest"))
    ci = conformal_intervals(r, alpha=0.1)
    assert ci["available"] and 0.84 <= ci["empirical_test_coverage"] <= 0.97
    iv = ci["intervals"]; assert (iv.upper > iv.lower).all()


def test_conformal_refuses_when_too_few_cal_rows_or_classification():
    r = train_model(reg_df(n=60), TrainConfig("y", "regression", "Linear Regression"))
    assert not conformal_intervals(r, alpha=0.01)["available"]
    c = train_model(clf_df(), TrainConfig("y", "classification", "Decision Tree"))
    assert not conformal_intervals(c)["available"]


def test_ood_flags_extreme_inputs_and_rate_is_controlled(monkeypatch):
    r = train_model(reg_df(n=1500), TrainConfig("y", "regression", "Random Forest"))
    import src.uncertainty as uncertainty
    monkeypatch.setattr(uncertainty, "MAX_OOD_FIT_ROWS", 100)
    monkeypatch.setattr(uncertainty, "MAX_OOD_CAL_ROWS", 100)
    det = fit_ood(r, 0.05)
    assert det.n_fit_ == min(100, len(r.X_fit)) and det.n_cal_ == min(100, len(r.X_cal))
    normal = det.score(r.X_test).is_out_of_distribution.mean()
    far = r.X_test.copy(); far["a"] = 50.0
    assert normal < 0.15 and det.score(far).is_out_of_distribution.mean() > 0.9
