import json

import numpy as np
import pandas as pd
import pytest

from src.monitoring import MonitorThresholds, acknowledge_alert, analyze_monitoring
from src.monitoring.reporting import build_report, csv_report_bytes, json_report_bytes
from src.monitoring.store import MonitoringStore


def _classification_batches():
    n = 90
    reference = pd.DataFrame({
        "feature": np.linspace(0, 1, n),
        "prediction": ["private-class-a" if i % 2 == 0 else "private-class-b" for i in range(n)],
        "label": ["private-class-a" if i % 2 == 0 else "private-class-b" for i in range(n)],
        "confidence": [0.95] * n,
        "latency_ms": np.linspace(2, 4, n),
    })
    current = reference.copy()
    current["feature"] = current["feature"] + 20
    current["confidence"] = [0.60] * n
    current["latency_ms"] = [25.0] * n
    current.loc[:44, "prediction"] = "private-class-b"
    return reference, current


def test_classification_monitoring_uses_supplied_batches_and_creates_real_alerts():
    reference, current = _classification_batches()
    thresholds = MonitorThresholds(p95_latency_max_ms=10)
    snapshot = analyze_monitoring(
        current, reference=reference, prediction_col="prediction", label_col="label",
        confidence_col="confidence", latency_col="latency_ms", task="classification",
        feature_columns=["feature"], thresholds=thresholds,
        current_fingerprint="current-hash", reference_fingerprint="reference-hash",
    )
    value = snapshot.to_dict()
    assert value["current_rows"] == value["reference_rows"] == 90
    assert value["performance"]["current"]["accuracy"] < value["performance"]["reference"]["accuracy"]
    assert value["drift_summary"]["summary"]["drift_features"] == ["feature"]
    assert value["latency_summary"]["p95_ms"] == 25
    assert {alert["category"] for alert in value["alerts"]} >= {
        "feature_drift", "confidence", "performance", "latency"}
    # Raw categorical labels are mapped to ordinal names in distributions and omitted from reports.
    assert "private-class-a" not in json_report_bytes([value]).decode()
    assert "private-class-b" not in json_report_bytes([value]).decode()


def test_unlabeled_batch_does_not_claim_performance_and_no_reference_has_no_drift():
    current = pd.DataFrame({"x": np.arange(40), "prediction": ["positive", "negative"] * 20})
    snapshot = analyze_monitoring(current, prediction_col="prediction", task="classification", feature_columns=["x"])
    assert snapshot.performance["current"] is None
    assert snapshot.performance["reference"] is None
    assert snapshot.drift_summary is None
    assert snapshot.current_rows == 40


def test_feature_schema_changes_are_reported_without_echoing_cell_values():
    reference = pd.DataFrame({"x": np.arange(40), "removed_feature": ["old-secret"] * 40,
                              "prediction": ["a", "b"] * 20})
    current = pd.DataFrame({"x": np.arange(40), "new_feature": ["new-secret"] * 40,
                            "prediction": ["a", "b"] * 20})
    snapshot = analyze_monitoring(current, reference=reference, prediction_col="prediction",
                                  feature_columns=["x", "removed_feature", "new_feature"])
    schema = snapshot.drift_summary["schema"]
    assert schema["missing_in_current"] == ["removed_feature"]
    assert schema["new_in_current"] == ["new_feature"]
    assert any(alert["category"] == "schema_change" for alert in snapshot.alerts)
    assert "old-secret" not in json_report_bytes([snapshot.to_dict()]).decode()
    assert "new-secret" not in json_report_bytes([snapshot.to_dict()]).decode()


def test_regression_monitoring_compares_measured_mae_and_rmse():
    n = 50
    reference = pd.DataFrame({"x": np.arange(n), "prediction": np.arange(n, dtype=float),
                              "target": np.arange(n, dtype=float)})
    current = reference.copy()
    current["prediction"] += 4
    thresholds = MonitorThresholds(error_increase_warning=0.1, error_increase_critical=0.25)
    snapshot = analyze_monitoring(current, reference=reference, prediction_col="prediction",
                                  label_col="target", task="regression", feature_columns=["x"],
                                  thresholds=thresholds)
    assert snapshot.performance["current"]["mae"] == pytest.approx(4)
    assert snapshot.performance["reference"]["mae"] == 0
    assert any(alert["category"] == "performance" for alert in snapshot.alerts)


def test_confidence_and_latency_invalid_values_are_explicitly_unavailable():
    frame = pd.DataFrame({"x": np.arange(35), "prediction": ["a", "b"] * 17 + ["a"],
                          "confidence": [1.2] * 35, "latency": [-1] * 35})
    snapshot = analyze_monitoring(frame, prediction_col="prediction", confidence_col="confidence",
                                  latency_col="latency", feature_columns=["x"])
    assert snapshot.confidence_summary["available"] is False
    assert snapshot.latency_summary["available"] is False


def test_monitoring_bounds_columns_and_thresholds_are_validated():
    with pytest.raises(ValueError, match="row"):
        analyze_monitoring(pd.DataFrame({"p": [1] * 200_001}), prediction_col="p")
    with pytest.raises(ValueError, match="Critical drift"):
        MonitorThresholds(drift_fraction_warning=.8, drift_fraction_critical=.5).validate()
    with pytest.raises(ValueError, match="Prediction column"):
        analyze_monitoring(pd.DataFrame({"x": [1]}), prediction_col="p")


def test_alert_acknowledgement_records_a_reviewer_label_and_timestamp():
    current = pd.DataFrame({"x": np.arange(40), "prediction": ["a", "b"] * 20,
                            "confidence": [.2] * 40})
    snapshot = analyze_monitoring(current, prediction_col="prediction", confidence_col="confidence",
                                  feature_columns=["x"])
    alert = snapshot.alerts[0]
    result = acknowledge_alert(snapshot.to_dict(), alert["alert_id"], "Reviewer 1")
    assert result["acknowledged"] and result["reviewer"] == "Reviewer 1"
    assert result["acknowledged_at"]
    with pytest.raises(ValueError):
        acknowledge_alert(snapshot.to_dict(), "missing", "Reviewer")


def test_report_excludes_raw_rows_and_escapes_formula_like_feature_names():
    current = pd.DataFrame({"=formula": np.arange(40), "prediction": ["secret-a", "secret-b"] * 20})
    snapshot = analyze_monitoring(current, prediction_col="prediction", feature_columns=["=formula"])
    report = build_report([snapshot.to_dict()])
    serialized = json.dumps(report)
    assert "secret-a" not in serialized and "secret-b" not in serialized
    assert "raw_rows" not in serialized and "payload" not in serialized
    csv_data = csv_report_bytes([snapshot.to_dict()]).decode()
    assert "'=formula" in csv_data
    assert json_report_bytes([snapshot.to_dict()])


def test_model_ood_is_optional_and_only_uses_matching_features():
    from helpers import clf_df
    from src.train import TrainConfig, train_model

    train = train_model(clf_df(n=350), TrainConfig("y", "classification", "Decision Tree"))
    current = train.X_test.copy().reset_index(drop=True)
    current["prediction"] = train.pipeline.predict(current)
    snapshot = analyze_monitoring(current, prediction_col="prediction", task="classification",
                                  feature_columns=list(train.features), model_result=train,
                                  model_source_fingerprint="trained-on-this-dataset")
    assert snapshot.model_version
    assert snapshot.ood_summary is not None
    assert snapshot.ood_summary.get("available") is True


def test_sqlite_store_is_session_scoped_persists_only_aggregate_reports_and_acknowledgements(tmp_path):
    reference, current = _classification_batches()
    snapshot = analyze_monitoring(current, prediction_col="prediction", reference=reference,
                                  label_col="label", confidence_col="confidence",
                                  feature_columns=["feature"], task="classification")
    store = MonitoringStore(tmp_path / "monitoring.sqlite3")
    scope_a = "11111111-1111-4111-8111-111111111111"
    scope_b = "22222222-2222-4222-8222-222222222222"
    store.add(scope_a, snapshot.to_dict())
    store.add(scope_b, snapshot.to_dict())
    loaded = store.list(scope_a)
    assert len(loaded) == 1 and store.list(scope_b)[0]["snapshot_id"] == snapshot.snapshot_id
    assert "private-class-a" not in json.dumps(loaded)
    assert "private-class-b" not in (tmp_path / "monitoring.sqlite3").read_bytes().decode("latin-1")
    if snapshot.alerts:
        alert = snapshot.alerts[0]
        updated = store.acknowledge(scope_a, snapshot.snapshot_id, alert["alert_id"], "Reviewer")
        assert next(item for item in updated["alerts"] if item["alert_id"] == alert["alert_id"])["acknowledged"]
        assert store.list(scope_b)[0]["alerts"][0]["acknowledged"] is False


def test_monitoring_page_renders_without_fabricated_events():
    pytest.importorskip("streamlit")
    from pathlib import Path
    from streamlit.testing.v1 import AppTest

    page = Path(__file__).resolve().parents[1] / "pages/9_Model_Monitoring.py"
    at = AppTest.from_file(page, default_timeout=60).run()
    assert not at.exception
    assert not at.session_state["monitoring_history"]
