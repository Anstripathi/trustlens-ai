import numpy as np
import pandas as pd

from helpers import clf_df, reg_df
from src.auditor.reliability_audit import audit_reliability
from src.auditor.schemas import AuditRequest, CheckStatus
from src.artifact_manager import dataset_fingerprint
from src.train import TrainConfig, train_model


def request(frame, model, **kwargs):
    fingerprint = dataset_fingerprint(frame)
    return AuditRequest(frame, "data.csv", fingerprint, model, fingerprint, **kwargs)


def test_classification_reliability_uses_calibration_and_test_splits():
    frame = clf_df(nan=False)
    model = train_model(frame, TrainConfig("y", "classification", "Logistic Regression"))
    checks = audit_reliability(request(frame, model))
    by_name = {check.check_id: check for check in checks}
    assert by_name["reliability.confidence"].status is CheckStatus.PASSED
    assert by_name["reliability.calibration"].status in (CheckStatus.PASSED, CheckStatus.FINDING)
    assert by_name["reliability.conformal"].status is CheckStatus.NOT_APPLICABLE
    assert by_name["reliability.ood"].status is CheckStatus.PASSED
    assert by_name["reliability.drift"].status is CheckStatus.NOT_EVALUATED


def test_regression_reliability_produces_conformal_interval_summary():
    frame = reg_df()
    model = train_model(frame, TrainConfig("y", "regression", "Linear Regression"))
    checks = audit_reliability(request(frame, model))
    by_name = {check.check_id: check for check in checks}
    assert by_name["reliability.confidence"].status is CheckStatus.NOT_APPLICABLE
    assert by_name["reliability.calibration"].status is CheckStatus.NOT_APPLICABLE
    assert by_name["reliability.conformal"].status in (CheckStatus.PASSED, CheckStatus.FINDING)


def test_drift_is_optional_and_uses_only_aggregate_counts():
    frame = pd.DataFrame({"score": np.arange(80, dtype=float), "group": ["a", "b"] * 40})
    changed = pd.DataFrame({"score": np.arange(80, dtype=float) + 100, "group": ["c", "d"] * 40})
    model_frame = clf_df(nan=False)
    model = train_model(model_frame, TrainConfig("y", "classification", "Decision Tree"))
    checks = audit_reliability(request(model_frame, model,
                                      reference_dataset=frame, current_dataset=changed))
    drift = next(check for check in checks if check.check_id == "reliability.drift")
    assert drift.status is CheckStatus.FINDING
    assert "features_with_drift" in drift.metrics
    assert "'c'" not in repr(drift.metrics)


def test_reliability_not_evaluated_when_disabled():
    checks = audit_reliability(AuditRequest(None, None, None, run_reliability=False))
    assert all(check.status is CheckStatus.NOT_EVALUATED for check in checks)
