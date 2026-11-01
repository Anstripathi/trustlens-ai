"""Manual model monitoring from user-supplied prediction batches."""
from __future__ import annotations

import hashlib
import uuid

import pandas as pd
import streamlit as st

from src import config
from src.artifact_manager import reject_model_artifact
from src.data_validation import ValidationError, load_csv_bytes
from src.monitoring import MonitorThresholds, acknowledge_alert, analyze_monitoring
from src.monitoring.reporting import csv_report_bytes, json_report_bytes
from src.monitoring.store import MonitoringStore
from src.theme import record_activity, render_page_header, render_sidebar

st.set_page_config(page_title="Model Monitoring", page_icon=":material/monitoring:", layout="wide")
render_sidebar()
render_page_header("Model Monitoring", "Inspect supplied prediction telemetry, compare batches, and review threshold-based alerts.")
st.info("This page has no production prediction API. Metrics below are computed only from CSV batches you provide; it does not create simulated monitoring events.")

try:
    monitor_store = MonitoringStore(config.ARTIFACTS_DIR / "monitoring.sqlite3")
except Exception:
    monitor_store = None
    st.caption("Local aggregate storage is unavailable; monitoring history will last only for this Streamlit session.")
if "monitoring_scope_id" not in st.session_state:
    st.session_state.monitoring_scope_id = str(uuid.uuid4())
if "monitoring_history" not in st.session_state:
    try:
        st.session_state.monitoring_history = monitor_store.list(st.session_state.monitoring_scope_id) if monitor_store else []
    except Exception:
        st.session_state.monitoring_history = []

with st.expander("Restore or back up local monitoring history"):
    st.caption("A recovery key is a bearer secret for aggregate records on this app installation. Keep it private. SQLite stores summary metrics only, not uploaded CSV rows; Community Cloud filesystems may be ephemeral.")
    recovery_key = st.text_input("Recovery key", type="pass" "word", key="monitor_recovery_key")
    if st.button("Restore history with key", disabled=monitor_store is None):
        try:
            scope_id = str(uuid.UUID(recovery_key.strip()))
            restored = monitor_store.list(scope_id)
            st.session_state.monitoring_scope_id = scope_id
            st.session_state.monitoring_history = restored
            st.session_state.pop("monitoring_latest_id", None)
            st.rerun()
        except (ValueError, TypeError, AttributeError):
            st.error("Enter a valid recovery key.")
        except Exception:
            st.error("Monitoring history could not be restored from local storage.")
    if monitor_store:
        st.download_button("Download recovery key", st.session_state.monitoring_scope_id,
                           "trustlens-monitoring-recovery-key.txt", "text/plain")

current_upload = st.file_uploader("Current prediction batch (CSV)", type=None, key="monitor_current_upload")
reference_upload = st.file_uploader("Optional reference prediction batch (CSV)", type=None, key="monitor_reference_upload")

current_frame = reference_frame = None
current_fp = reference_fp = None
try:
    if current_upload is not None:
        reject_model_artifact(current_upload.name)
        loaded = load_csv_bytes(current_upload.getvalue(), current_upload.name)
        current_frame = loaded.clean
        current_fp = hashlib.sha256(current_upload.getvalue()).hexdigest()[:16]
    if reference_upload is not None:
        reject_model_artifact(reference_upload.name)
        loaded_reference = load_csv_bytes(reference_upload.getvalue(), reference_upload.name)
        reference_frame = loaded_reference.clean
        reference_fp = hashlib.sha256(reference_upload.getvalue()).hexdigest()[:16]
except ValidationError as exc:
    st.error(str(exc))
    st.stop()

if current_frame is None:
    st.info("Upload a current prediction CSV to begin. Include predicted labels or values; confidence, true labels, and measured latency are optional columns.")
    st.stop()

def _column_select(label: str, frame: pd.DataFrame, key: str, *, optional: bool = True):
    values = list(frame.columns)
    options = (["(none)"] if optional else []) + values
    label_lower = label.lower()
    candidates = (("prediction", "predicted", "predicted_label", "y_pred") if "prediction" in label_lower else
                 ("true_label", "label", "target") if "label" in label_lower else
                 ("confidence", "probability") if "confidence" in label_lower else
                 ("latency_ms",) if "latency" in label_lower else ())
    default = next((name for name in candidates if name in values), None)
    index = (options.index(default) if default in options else 0)
    return st.selectbox(label, options, index=index, key=key)

st.caption(f"Current batch: {len(current_frame):,} rows · {current_frame.shape[1]} columns · file fingerprint `{current_fp}`")
task = st.radio("Prediction task", ["classification", "regression"], horizontal=True)
prediction_col = _column_select("Prediction column", current_frame, "monitor_prediction", optional=False)
label_choice = _column_select("Optional true-label column", current_frame, "monitor_label")
confidence_choice = _column_select("Optional confidence column (0–1)", current_frame, "monitor_confidence")
latency_choice = _column_select("Optional measured latency column (milliseconds)", current_frame, "monitor_latency")
exclude = {prediction_col, label_choice, confidence_choice, latency_choice, "(none)"}
default_features = [name for name in current_frame.columns if name not in exclude and name.lower() not in {
    "prediction", "predicted", "predicted_label", "y_pred", "confidence", "probability", "latency_ms",
    "true_label", "label", "target"}]
feature_choices = list(current_frame.columns)
if reference_frame is not None:
    feature_choices = list(dict.fromkeys([*current_frame.columns, *reference_frame.columns]))
feature_choices = [name for name in feature_choices if name not in exclude]
feature_columns = st.multiselect("Feature columns to compare and summarize", feature_choices,
                                 default=default_features)

reference_prediction_ready = reference_frame is not None and prediction_col in reference_frame.columns
if reference_frame is not None:
    st.caption(f"Reference batch: {len(reference_frame):,} rows · {reference_frame.shape[1]} columns · file fingerprint `{reference_fp}`")
    if not reference_prediction_ready:
        st.warning("The selected prediction column is missing in the reference file; prediction distributions and performance comparison will be unavailable.")

active_model = st.session_state.get("result")
active_dataset_fp = st.session_state.get("df_fp")
model_dataset_fp = st.session_state.get("model_data_fp")
active_model_current = active_model is not None and active_dataset_fp and model_dataset_fp == active_dataset_fp
use_ood = st.checkbox("Compare against the active model's fit/calibration distribution (OOD)",
                      value=False, disabled=not active_model_current)
if not active_model_current:
    st.caption("Train a current model in Model Lab to enable the optional OOD comparison.")

with st.expander("Alert thresholds"):
    drift_warning = st.slider("Drifted feature fraction warning", 0.0, 1.0, 0.10, 0.01)
    drift_critical = st.slider("Drifted feature fraction critical", drift_warning, 1.0, max(0.30, drift_warning), 0.01)
    ood_warning = st.slider("OOD fraction warning", 0.0, 1.0, 0.10, 0.01)
    ood_critical = st.slider("OOD fraction critical", ood_warning, 1.0, max(0.25, ood_warning), 0.01)
    confidence_min = st.slider("Mean confidence warning threshold", 0.0, 1.0, 0.55, 0.01)
    accuracy_drop = st.slider("Accuracy-drop warning threshold", 0.0, 1.0, 0.05, 0.01)
    error_increase = st.slider("Regression error increase warning threshold", 0.0, 1.0, 0.10, 0.01)
    ece_increase = st.slider("Top-label ECE increase warning threshold", 0.0, 1.0, 0.05, 0.01)
    use_latency = st.checkbox("Alert on p95 measured latency")
    latency_limit = st.number_input("p95 latency limit (ms)", min_value=0.0, max_value=86_400_000.0,
                                    value=500.0, disabled=not use_latency)

run = st.button("Analyze supplied monitoring batch", type="primary")
if run:
    try:
        thresholds = MonitorThresholds(
            drift_fraction_warning=drift_warning, drift_fraction_critical=drift_critical,
            ood_fraction_warning=ood_warning, ood_fraction_critical=ood_critical,
            confidence_mean_min=confidence_min, accuracy_drop_warning=accuracy_drop,
            accuracy_drop_critical=max(accuracy_drop, 0.15),
            error_increase_warning=error_increase,
            error_increase_critical=max(error_increase, 0.25),
            ece_increase_warning=ece_increase,
            p95_latency_max_ms=float(latency_limit) if use_latency else None,
        )
        label_col = None if label_choice == "(none)" else label_choice
        confidence_col = None if confidence_choice == "(none)" else confidence_choice
        latency_col = None if latency_choice == "(none)" else latency_choice
        model_for_run = active_model if use_ood else None
        snapshot = analyze_monitoring(
            current_frame, prediction_col=prediction_col, reference=reference_frame, task=task,
            label_col=label_col, confidence_col=confidence_col, latency_col=latency_col,
            feature_columns=feature_columns, model_result=model_for_run, thresholds=thresholds,
            current_fingerprint=current_fp, reference_fingerprint=reference_fp,
            model_source_fingerprint=st.session_state.get("model_data_fp") if model_for_run else None,
        )
        snapshot_data = snapshot.to_dict()
        if monitor_store:
            try:
                monitor_store.add(st.session_state.monitoring_scope_id, snapshot_data)
                st.session_state.monitoring_history = monitor_store.list(st.session_state.monitoring_scope_id)
            except Exception:
                st.session_state.monitoring_history.insert(0, snapshot_data)
                st.session_state.monitoring_history = st.session_state.monitoring_history[:100]
                st.warning("The snapshot ran successfully, but local persistence failed; it remains in this session only.")
        else:
            st.session_state.monitoring_history.insert(0, snapshot_data)
            st.session_state.monitoring_history = st.session_state.monitoring_history[:100]
        record_activity("Monitoring batch analyzed", f"{snapshot.current_rows:,} supplied rows · {len(snapshot.alerts)} alert(s)")
        st.session_state.monitoring_latest_id = snapshot.snapshot_id
        st.rerun()
    except (ValueError, TypeError) as exc:
        st.error(str(exc))
    except Exception:
        st.error("Monitoring could not be completed. Check the selected columns and try again.")

history = st.session_state.monitoring_history
latest_id = st.session_state.get("monitoring_latest_id")
latest = next((item for item in history if item["snapshot_id"] == latest_id), history[0] if history else None)
if latest is not None:
    st.divider()
    st.subheader("Latest monitoring snapshot")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Usable predictions", f"{latest['prediction_count']:,}")
    c2.metric("Reference rows", "Not supplied" if latest["reference_rows"] is None else f"{latest['reference_rows']:,}")
    c3.metric("Alerts", len(latest["alerts"]))
    c4.metric("Snapshot status", latest["status"].title())
    st.caption(f"Snapshot {latest['snapshot_id']} · {latest['created_at']} · task: {latest['task']}")
    st.info(latest["health_summary"])

    st.markdown("**Prediction distribution**")
    prediction_summary = latest["prediction_summary"]
    if prediction_summary.get("kind") == "categorical":
        classes = prediction_summary.get("classes", [])
        if classes:
            st.bar_chart(pd.DataFrame(classes).set_index("class"))
        if prediction_summary.get("comparison"):
            st.json(prediction_summary["comparison"])
    else:
        st.json(prediction_summary)

    st.markdown("**Labeled performance**")
    performance = latest.get("performance") or {}
    if performance.get("current") is None:
        st.info("No usable true-label column was supplied; performance is not evaluated.")
    else:
        st.json(performance)

    if latest.get("confidence_summary"):
        st.markdown("**Confidence distribution**")
        st.json(latest["confidence_summary"])
    if latest.get("latency_summary"):
        st.markdown("**Measured latency**")
        st.json(latest["latency_summary"])
    if latest.get("drift_summary"):
        st.markdown("**Feature drift**")
        drift = latest["drift_summary"]
        if drift.get("available"):
            st.dataframe(pd.DataFrame(drift["features"]), width="stretch", hide_index=True)
            st.caption(drift["disclaimer"])
        else:
            st.info(drift.get("reason", "Drift was not evaluated."))
    if latest.get("ood_summary"):
        st.markdown("**Out-of-distribution comparison**")
        st.json(latest["ood_summary"])
    if latest.get("feature_statistics"):
        st.markdown("**Current feature statistics**")
        st.dataframe(pd.DataFrame(latest["feature_statistics"]).T, width="stretch")

    if latest["alerts"]:
        st.markdown("**Alerts and acknowledgement**")
        st.dataframe(pd.DataFrame(latest["alerts"])[["severity", "category", "message", "recommendation", "acknowledged", "reviewer"]],
                     width="stretch", hide_index=True)
        pending_alerts = [item for item in latest["alerts"] if not item["acknowledged"]]
        if pending_alerts:
            selected_alert = st.selectbox("Alert to acknowledge", pending_alerts,
                                          format_func=lambda item: f"{item['severity']} · {item['category']} · {item['alert_id'][:8]}")
            reviewer = st.text_input("Reviewer label", value="Session reviewer", max_chars=64)
            if st.button("Acknowledge selected alert"):
                try:
                    if monitor_store:
                        updated = monitor_store.acknowledge(st.session_state.monitoring_scope_id,
                                                           latest["snapshot_id"], selected_alert["alert_id"], reviewer)
                        st.session_state.monitoring_history = [updated if item["snapshot_id"] == updated["snapshot_id"] else item
                                                              for item in st.session_state.monitoring_history]
                    else:
                        acknowledge_alert(latest, selected_alert["alert_id"], reviewer)
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))
    else:
        st.success("No configured thresholds were exceeded in the supplied batch.")
        st.caption("No alerts does not establish model quality or safety.")

st.subheader("Monitoring history")
if history:
    st.dataframe([{"Created": item["created_at"], "Snapshot": item["snapshot_id"],
                   "Task": item["task"], "Current rows": item["current_rows"],
                   "Usable predictions": item["prediction_count"],
                   "Reference rows": item["reference_rows"] or "Not supplied",
                   "Current data version": item.get("dataset_versions", {}).get("current") or "Not recorded",
                   "Reference data version": item.get("dataset_versions", {}).get("reference") or "Not recorded",
                   "Alerts": len(item["alerts"]), "Model version": item["model_version"] or "Not assessed"}
                  for item in history], width="stretch", hide_index=True)
    json_bytes = json_report_bytes(history)
    csv_bytes = csv_report_bytes(history)
    export_cols = st.columns(2)
    export_cols[0].download_button("Download monitoring history (JSON)", json_bytes,
                                   "trustlens-monitoring-history.json", "application/json", width="stretch")
    export_cols[1].download_button("Download monitoring history (CSV)", csv_bytes,
                                   "trustlens-monitoring-history.csv", "text/csv", width="stretch")
else:
    st.caption("History will contain snapshots only after you analyze a supplied batch. It is session-local and resets with the Streamlit session.")
