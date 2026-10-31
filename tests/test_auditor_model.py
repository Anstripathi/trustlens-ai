from dataclasses import replace

import pandas as pd

from helpers import clf_df, reg_df
from src.auditor import model_audit
from src.auditor.model_audit import audit_model
from src.auditor.schemas import AuditRequest, CheckStatus
from src.artifact_manager import dataset_fingerprint
from src.train import TrainConfig, train_model


def request(frame, model, **kwargs):
    fingerprint = dataset_fingerprint(frame)
    return AuditRequest(frame, "dataset.csv", fingerprint, model=model,
                        model_fingerprint=fingerprint, **kwargs)


def test_model_audit_uses_test_metrics_and_preserves_active_pipeline():
    frame = clf_df(nan=False)
    model = train_model(frame, TrainConfig("y", "classification", "Logistic Regression"))
    before = model.pipeline.predict(model.X_test)
    checks = audit_model(request(frame, model))
    performance = next(item for item in checks if item.check_id == "model.performance")
    split = next(item for item in checks if item.check_id == "model.split_integrity")
    assert performance.status in (CheckStatus.PASSED, CheckStatus.FINDING)
    assert performance.metrics["test_rows"] == len(model.X_test)
    assert set(performance.metrics["classwise_test_metrics"]) == {"class_1", "class_2"}
    assert len(performance.metrics["confusion_matrix"]["matrix"]) == 2
    assert split.metrics["test_rows"] == len(model.X_test)
    assert (model.pipeline.predict(model.X_test) == before).all()


def test_model_audit_reports_no_model_and_stale_model_as_not_evaluated():
    frame = clf_df(nan=False)
    fingerprint = dataset_fingerprint(frame)
    absent = audit_model(AuditRequest(frame, "x", fingerprint))
    assert all(check.status is CheckStatus.NOT_EVALUATED for check in absent)
    model = train_model(frame, TrainConfig("y", "classification", "Decision Tree"))
    stale = audit_model(AuditRequest(frame, "x", fingerprint, model, "different"))
    assert all(check.status is CheckStatus.NOT_EVALUATED for check in stale)


def test_regression_model_metrics_include_rmse_and_r2():
    frame = reg_df()
    model = train_model(frame, TrainConfig("y", "regression", "Linear Regression"))
    performance = next(check for check in audit_model(request(frame, model))
                       if check.check_id == "model.performance")
    assert "rmse" in performance.metrics["test_metrics"]
    assert "r2" in performance.metrics["test_metrics"]


def test_validation_benchmark_is_explicit_and_does_not_replace_model(monkeypatch):
    frame = clf_df(nan=False)
    model = train_model(frame, TrainConfig("y", "classification", "Decision Tree"))
    pipeline = model.pipeline
    seen = {}

    def fake_compare(dataset, config):
        seen["dataset"] = dataset
        return pd.DataFrame({"f1": [0.7]}, index=["Decision Tree"])

    monkeypatch.setattr(model_audit, "compare_models", fake_compare)
    checks = audit_model(request(frame, model, benchmark_models=True))
    benchmark = next(check for check in checks if check.check_id == "model.validation_comparison")
    assert benchmark.status is CheckStatus.PASSED
    assert benchmark.metrics["selection_split"] == "calibration/validation"
    assert benchmark.metrics["active_model_changed"] is False
    assert seen["dataset"] is frame
    assert model.pipeline is pipeline


def test_overlapping_split_indices_are_reported():
    frame = clf_df(nan=False)
    model = train_model(frame, TrainConfig("y", "classification", "Decision Tree"))
    overlapping = replace(model, X_test=model.X_fit.iloc[:2], y_test=model.y_fit.iloc[:2])
    split = next(check for check in audit_model(request(frame, overlapping))
                 if check.check_id == "model.split_integrity")
    assert split.status is CheckStatus.FINDING
    assert split.findings[0].title == "Split row indices overlap"


def test_optional_cross_validation_refits_pipeline_clones_on_fit_only():
    frame = clf_df(n=240, nan=False)
    model = train_model(frame, TrainConfig("y", "classification", "Logistic Regression"))
    pipeline = model.pipeline
    cross_validation = next(check for check in audit_model(request(frame, model, run_cross_validation=True))
                            if check.check_id == "model.cross_validation")
    assert cross_validation.status is CheckStatus.PASSED
    assert cross_validation.metrics["folds"] == 3
    assert cross_validation.metrics["rows"] == len(model.X_fit)
    assert model.pipeline is pipeline
