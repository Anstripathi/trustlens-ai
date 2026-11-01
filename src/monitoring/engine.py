"""Manual prediction and feature monitoring with bounded, aggregate-only results."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib

import numpy as np
import pandas as pd
from sklearn import metrics

from src import config
from src.drift import compare_schemas, drift_report
from src.uncertainty import fit_ood, MAX_OOD_CAL_ROWS

from .schemas import MonitorThresholds, MonitoringSnapshot, create_alert

MAX_MONITOR_ROWS = 200_000
MAX_DRIFT_ROWS = 10_000
MAX_CATEGORICAL_LEVELS = 20
_META_COLUMNS = {"prediction", "predicted", "label", "true_label", "target", "confidence", "probability", "latency_ms"}


def _sample(frame: pd.DataFrame, limit: int, seed: int = 42) -> pd.DataFrame:
    if len(frame) <= limit:
        return frame
    return frame.sample(limit, random_state=seed)


def _validate_frame(frame: pd.DataFrame, name: str) -> None:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(f"{name} prediction data must be a non-empty DataFrame.")
    if len(frame) > MAX_MONITOR_ROWS:
        raise ValueError(f"{name} data exceeds the {MAX_MONITOR_ROWS:,}-row monitoring limit.")
    if not 1 <= frame.shape[1] <= config.MAX_COLS:
        raise ValueError("Monitoring data has an unsupported number of columns.")


def _summary_stats(series: pd.Series) -> dict:
    missing = int(series.isna().sum())
    base = {"rows": int(len(series)), "missing": missing,
            "missing_fraction": float(missing / len(series)) if len(series) else 0.0}
    if pd.api.types.is_numeric_dtype(series):
        values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if values.empty:
            return {**base, "kind": "numeric", "valid": 0}
        return {**base, "kind": "numeric", "valid": int(len(values)),
                "mean": float(values.mean()), "std": float(values.std(ddof=0)),
                "min": float(values.min()), "median": float(values.median()),
                "p95": float(values.quantile(.95)), "max": float(values.max())}
    values = series.dropna().astype(str)
    return {**base, "kind": "categorical", "valid": int(len(values)),
            "distinct_count": int(values.nunique())}


def _distribution(frame: pd.DataFrame, column: str, task: str,
                  shared_labels: list[str] | None = None) -> dict:
    if task == "regression":
        values = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if values.empty:
            return {"kind": "numeric", "valid": 0, "quantiles": None}
        return {"kind": "numeric", "valid": int(len(values)), "quantiles": {
            "p05": float(values.quantile(.05)), "p25": float(values.quantile(.25)),
            "median": float(values.median()), "p75": float(values.quantile(.75)),
            "p95": float(values.quantile(.95))}}
    counts = frame[column].dropna().astype(str).value_counts()
    labels = shared_labels if shared_labels is not None else sorted(counts.index)[:MAX_CATEGORICAL_LEVELS]
    rows = [{"class": f"Class {index + 1}", "count": int(counts.get(label, 0))}
            for index, label in enumerate(labels[:MAX_CATEGORICAL_LEVELS])]
    other_count = int(counts.drop(index=labels, errors="ignore").sum())
    if other_count:
        rows.append({"class": "Other", "count": other_count})
    return {"kind": "categorical", "valid": int(counts.sum()), "classes": rows}


def _performance(frame: pd.DataFrame | None, *, task: str, prediction_col: str,
                 label_col: str | None, confidence_col: str | None) -> dict | None:
    if frame is None or label_col is None or label_col not in frame or prediction_col not in frame:
        return None
    frame = frame.reset_index(drop=True)
    paired = frame[[label_col, prediction_col]].dropna()
    if paired.empty:
        return None
    y_true, y_pred = paired[label_col], paired[prediction_col]
    if task == "classification":
        labels = sorted(set(y_true.astype(str)) | set(y_pred.astype(str)))
        yt, yp = y_true.astype(str), y_pred.astype(str)
        result = {"n_labeled": int(len(paired)),
                  "accuracy": float(metrics.accuracy_score(yt, yp)),
                  "precision_macro": float(metrics.precision_score(yt, yp, average="macro", zero_division=0)),
                  "recall_macro": float(metrics.recall_score(yt, yp, average="macro", zero_division=0)),
                  "f1_macro": float(metrics.f1_score(yt, yp, average="macro", zero_division=0))}
        if confidence_col and confidence_col in frame:
            conf = pd.to_numeric(frame.loc[paired.index, confidence_col], errors="coerce")
            valid = conf.notna() & conf.between(0, 1)
            if valid.any():
                correct = (yt.loc[valid] == yp.loc[valid]).astype(float).to_numpy()
                scores = conf.loc[valid].to_numpy(dtype=float)
                bins = np.linspace(0, 1, 11)
                ece = 0.0
                for left, right in zip(bins[:-1], bins[1:]):
                    selected = (scores >= left) & ((scores < right) if right < 1 else (scores <= right))
                    if selected.any():
                        ece += float(selected.mean()) * abs(float(scores[selected].mean()) - float(correct[selected].mean()))
                result["ece_top_label"] = float(ece)
                result["confidence_rows"] = int(valid.sum())
        return result
    yt = pd.to_numeric(y_true, errors="coerce")
    yp = pd.to_numeric(y_pred, errors="coerce")
    valid = yt.notna() & yp.notna() & np.isfinite(yt) & np.isfinite(yp)
    if not valid.any():
        return None
    yt, yp = yt[valid], yp[valid]
    mse = float(metrics.mean_squared_error(yt, yp))
    return {"n_labeled": int(len(yt)), "mae": float(metrics.mean_absolute_error(yt, yp)),
            "rmse": float(np.sqrt(mse)),
            "r2": float(metrics.r2_score(yt, yp)) if len(yt) > 1 else None}


def _distribution_drift(reference: pd.DataFrame, current: pd.DataFrame, *,
                        prediction_col: str, task: str) -> dict | None:
    if prediction_col not in reference or prediction_col not in current:
        return None
    if task == "classification":
        labels = sorted(set(reference[prediction_col].dropna().astype(str)) |
                        set(current[prediction_col].dropna().astype(str)))[:MAX_CATEGORICAL_LEVELS]
        return {"reference": _distribution(reference, prediction_col, task, labels),
                "current": _distribution(current, prediction_col, task, labels)}
    return {"reference": _distribution(reference, prediction_col, task),
            "current": _distribution(current, prediction_col, task)}


def _confidence_summary(frame: pd.DataFrame, column: str | None) -> dict | None:
    if not column or column not in frame:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return {"available": False, "reason": "The selected confidence column had no numeric values."}
    valid = values.between(0, 1)
    if not valid.all():
        return {"available": False, "reason": "Confidence must be numeric values from 0 to 1."}
    return {"available": True, "rows": int(len(values)), "mean": float(values.mean()),
            "median": float(values.median()), "p05": float(values.quantile(.05)),
            "p95": float(values.quantile(.95)), "min": float(values.min()), "max": float(values.max())}


def _latency_summary(frame: pd.DataFrame, column: str | None) -> dict | None:
    if not column or column not in frame:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty or (values < 0).any():
        return {"available": False, "reason": "Latency requires non-negative numeric milliseconds."}
    return {"available": True, "rows": int(len(values)), "mean_ms": float(values.mean()),
            "median_ms": float(values.median()), "p95_ms": float(values.quantile(.95)),
            "max_ms": float(values.max())}


def _drift(reference: pd.DataFrame | None, current: pd.DataFrame, columns: list[str]) -> dict | None:
    if reference is None:
        return None
    if not columns:
        return {"available": False, "reason": "No feature columns were selected for drift analysis."}
    ref_cols = [name for name in columns if name in reference]
    cur_cols = [name for name in columns if name in current]
    ref_subset, cur_subset = reference[ref_cols], current[cur_cols]
    schema = compare_schemas(ref_subset, cur_subset)
    common = schema["common"]
    if not common:
        return {"available": False, "reason": "Reference and current data have no selected feature columns in common.",
                "schema": schema}
    ref_sample, cur_sample = _sample(reference[common], MAX_DRIFT_ROWS), _sample(current[common], MAX_DRIFT_ROWS)
    report = drift_report(ref_sample, cur_sample)
    table = report["table"]
    if table.empty:
        rows = []
    else:
        allowed = [name for name in ("feature", "type", "status", "test", "p_adj", "effect_size",
                                      "effect_name", "missing_ref_%", "missing_cur_%") if name in table]
        rows = table[allowed].where(pd.notna(table[allowed]), None).to_dict("records")
    summary = report["summary"]
    return {"available": True, "summary": summary, "features": rows, "schema": schema,
            "sampled_reference_rows": int(len(ref_sample)), "sampled_current_rows": int(len(cur_sample)),
            "disclaimer": report["disclaimer"]}


def _add_alerts(*, alerts: list[dict], thresholds: MonitorThresholds,
                drift: dict | None, confidence: dict | None,
                reference_confidence: dict | None,
                performance: dict | None, reference_performance: dict | None,
                latency: dict | None, ood: dict | None) -> None:
    if drift:
        schema = drift.get("schema", {})
        if schema.get("missing_in_current") or schema.get("new_in_current") or schema.get("type_mismatch"):
            alerts.append(create_alert("warning", "schema_change", "Reference and current feature schemas differ.", "Review missing, new, and type-changed features before comparing predictions."))
    if drift and drift.get("available"):
        summary = drift["summary"]
        tested = max(1, summary.get("tested", 0))
        ratio = len(summary.get("drift_features", [])) / tested
        if ratio >= thresholds.drift_fraction_critical:
            alerts.append(create_alert("critical", "feature_drift", f"{ratio:.0%} of tested features exceeded practical drift thresholds.", "Review recent data sources and model assumptions before relying on predictions."))
        elif ratio >= thresholds.drift_fraction_warning:
            alerts.append(create_alert("warning", "feature_drift", f"{ratio:.0%} of tested features exceeded practical drift thresholds.", "Inspect the affected feature statistics and investigate changes in data collection."))
    if confidence and confidence.get("available") and confidence["mean"] < thresholds.confidence_mean_min:
        alerts.append(create_alert("warning", "confidence", "Mean supplied prediction confidence is below the configured threshold.", "Review low-confidence cases and verify that confidence values are calibrated for this model."))
    if confidence and confidence.get("available") and reference_confidence and reference_confidence.get("available"):
        drop = reference_confidence["mean"] - confidence["mean"]
        if drop >= thresholds.confidence_drop_warning:
            alerts.append(create_alert("warning", "confidence", f"Mean supplied confidence decreased by {drop:.3f} against the reference batch.", "Review changes in score distributions and calibration on recent labeled data."))
    if performance and reference_performance and "accuracy" in performance and "accuracy" in reference_performance:
        for metric_name in ("accuracy", "precision_macro", "recall_macro", "f1_macro"):
            drop = reference_performance[metric_name] - performance[metric_name]
            if drop >= thresholds.accuracy_drop_critical:
                alerts.append(create_alert("critical", "performance", f"{metric_name} decreased by {drop:.1%} against the supplied reference batch.", "Validate labels and data quality; consider retraining only after a controlled evaluation."))
            elif drop >= thresholds.accuracy_drop_warning:
                alerts.append(create_alert("warning", "performance", f"{metric_name} decreased by {drop:.1%} against the supplied reference batch.", "Collect more labeled examples and inspect class-wise errors."))
    if performance and reference_performance:
        for metric_name in ("mae", "rmse"):
            if metric_name not in performance or metric_name not in reference_performance:
                continue
            baseline = reference_performance[metric_name]
            measured = performance[metric_name]
            if abs(baseline) <= 1e-12:
                if measured > baseline:
                    alerts.append(create_alert("critical", "performance", f"{metric_name.upper()} increased from zero to {measured:.4g} against the supplied reference batch.", "Check label quality and feature distributions before accepting current predictions."))
                continue
            increase = (measured - baseline) / abs(baseline)
            if increase >= thresholds.error_increase_critical:
                alerts.append(create_alert("critical", "performance", f"{metric_name.upper()} increased by {increase:.1%} against the supplied reference batch.", "Check label quality and feature distributions before accepting current predictions."))
            elif increase >= thresholds.error_increase_warning:
                alerts.append(create_alert("warning", "performance", f"{metric_name.upper()} increased by {increase:.1%} against the supplied reference batch.", "Inspect error trends and collect additional labeled data."))
    if performance and reference_performance and "ece_top_label" in performance and "ece_top_label" in reference_performance:
        delta = performance["ece_top_label"] - reference_performance["ece_top_label"]
        if delta >= thresholds.ece_increase_warning:
            alerts.append(create_alert("warning", "calibration", f"Top-label ECE increased by {delta:.3f} against the reference batch.", "Recheck calibration on a representative labeled calibration set."))
    if latency and latency.get("available") and thresholds.p95_latency_max_ms is not None:
        if latency["p95_ms"] > thresholds.p95_latency_max_ms:
            alerts.append(create_alert("warning", "latency", "Measured p95 latency exceeds the configured threshold.", "Review serving hardware and measure latency under representative load."))
    if ood and ood.get("available"):
        fraction = ood["flagged_fraction"]
        if fraction >= thresholds.ood_fraction_critical:
            alerts.append(create_alert("critical", "out_of_distribution", f"{fraction:.0%} of sampled inputs were flagged as unusual by the active model's OOD detector.", "Check schema and input drift; unusual inputs can invalidate model assumptions."))
        elif fraction >= thresholds.ood_fraction_warning:
            alerts.append(create_alert("warning", "out_of_distribution", f"{fraction:.0%} of sampled inputs were flagged as unusual by the active model's OOD detector.", "Review flagged rates and compare current inputs with the model's fit distribution."))


def analyze_monitoring(current: pd.DataFrame, *, prediction_col: str,
                       reference: pd.DataFrame | None = None, task: str = "classification",
                       label_col: str | None = None, confidence_col: str | None = None,
                       latency_col: str | None = None, feature_columns: list[str] | None = None,
                       model_result=None, thresholds: MonitorThresholds | None = None,
                       current_fingerprint: str | None = None,
                       reference_fingerprint: str | None = None,
                       model_source_fingerprint: str | None = None) -> MonitoringSnapshot:
    """Create aggregate-only telemetry from user-supplied prediction CSV data."""
    _validate_frame(current, "Current")
    if reference is not None:
        _validate_frame(reference, "Reference")
    if task not in ("classification", "regression"):
        raise ValueError("Task must be classification or regression.")
    if prediction_col not in current:
        raise ValueError("Prediction column must exist in the current batch.")
    if label_col and label_col not in current:
        raise ValueError("Label column must exist in the current batch.")
    if confidence_col and confidence_col not in current:
        raise ValueError("Confidence column is unavailable in the current batch.")
    if latency_col and latency_col not in current:
        raise ValueError("Latency column is unavailable in the current batch.")
    thresholds = (thresholds or MonitorThresholds()).validate()

    excluded = {prediction_col}
    excluded.update({col for col in (label_col, confidence_col, latency_col) if col})
    default_features = [name for name in current.columns if name not in excluded and name.lower() not in _META_COLUMNS]
    selected_features = default_features if feature_columns is None else list(feature_columns)
    available_features = set(current.columns) | (set(reference.columns) if reference is not None else set())
    if set(selected_features) & excluded or set(selected_features) - available_features:
        raise ValueError("Feature selection contains an unknown or non-feature column.")

    current_performance = _performance(current, task=task, prediction_col=prediction_col,
                                       label_col=label_col, confidence_col=confidence_col)
    reference_performance = _performance(reference, task=task, prediction_col=prediction_col,
                                         label_col=label_col, confidence_col=confidence_col) if reference is not None else None
    confidence = _confidence_summary(current, confidence_col)
    latency = _latency_summary(current, latency_col)
    prediction_summary = _distribution(current, prediction_col, task)
    if reference is not None:
        prediction_summary["comparison"] = _distribution_drift(
            reference, current, prediction_col=prediction_col, task=task)

    feature_statistics = {column: _summary_stats(current[column]) for column in selected_features
                          if column in current.columns}
    drift = None
    if reference is not None:
        drift = _drift(reference, current, selected_features)

    ood_summary = None
    model_version = None
    if model_result is not None:
        model_meta = {"task": getattr(model_result, "task", None),
                      "model": getattr(getattr(model_result, "config", None), "model_name", None),
                      "seed": getattr(getattr(model_result, "config", None), "seed", None),
                      "hyperparams": getattr(getattr(model_result, "config", None), "hyperparams", None),
                      "features": list(getattr(model_result, "features", []))}
        model_meta["source_fingerprint"] = model_source_fingerprint
        model_version = hashlib.sha256(repr(model_meta).encode("utf-8")).hexdigest()[:16]
        features = model_meta["features"]
        if features and set(features).issubset(current.columns) and getattr(model_result, "task", None) == task:
            try:
                scored_rows = _sample(current[features], MAX_OOD_CAL_ROWS)
                detector = fit_ood(model_result)
                scores = detector.score(scored_rows)
                flagged = int(scores["is_out_of_distribution"].sum())
                ood_summary = {"available": True, "sampled_rows": int(len(scores)),
                               "flagged_rows": flagged, "flagged_fraction": float(flagged / len(scores)),
                               "threshold": float(detector.threshold_), "notes": detector.notes}
            except Exception:
                ood_summary = {"available": False, "reason": "The active model could not produce an OOD comparison for these features."}
        else:
            ood_summary = {"available": False, "reason": "Active model task or feature schema does not match the supplied current batch."}

    alerts: list[dict] = []
    reference_confidence = _confidence_summary(reference, confidence_col) if reference is not None else None
    _add_alerts(alerts=alerts, thresholds=thresholds, drift=drift, confidence=confidence,
                reference_confidence=reference_confidence,
                performance=current_performance, reference_performance=reference_performance,
                latency=latency, ood=ood_summary)
    versions = {"current": current_fingerprint, "reference": reference_fingerprint}
    health_summary = ("Critical alerts require review" if any(a["severity"] == "critical" for a in alerts)
                      else "Warning alerts require review" if alerts
                      else "No configured thresholds exceeded in the supplied evidence; checks may be incomplete")
    return MonitoringSnapshot(
        created_at=datetime.now(timezone.utc).isoformat(), current_rows=int(len(current)),
        prediction_count=int(prediction_summary.get("valid", 0)),
        reference_rows=int(len(reference)) if reference is not None else None, task=task,
        prediction_summary=prediction_summary, confidence_summary=confidence,
        feature_statistics=feature_statistics, drift_summary=drift,
        performance={"current": current_performance, "reference": reference_performance,
                     "comparison_available": bool(current_performance and reference_performance)},
        latency_summary=latency, ood_summary=ood_summary, model_version=model_version,
        dataset_versions=versions, health_summary=health_summary, alerts=alerts,
        status="partial" if (drift and not drift.get("available")) or (confidence and not confidence.get("available")) else "complete",
    )


def acknowledge_alert(snapshot: dict, alert_id: str, reviewer: str) -> dict:
    """Acknowledge an alert in a session-owned snapshot; reviewer is an unauthenticated label."""
    if not isinstance(reviewer, str) or not reviewer.strip() or len(reviewer.strip()) > 64:
        raise ValueError("A reviewer label of 1–64 characters is required.")
    alert = next((item for item in snapshot.get("alerts", []) if item.get("alert_id") == alert_id), None)
    if alert is None:
        raise ValueError("Alert was not found in the monitoring history.")
    alert["acknowledged"] = True
    alert["reviewer"] = reviewer.strip()
    alert["acknowledged_at"] = datetime.now(timezone.utc).isoformat()
    return alert
