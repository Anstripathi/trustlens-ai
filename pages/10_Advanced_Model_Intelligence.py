"""Experiment tracking and safe metadata-only model versioning."""
import time

import streamlit as st

from src import config
from src.evaluate import evaluate
from src.experiments import ExperimentRegistry, experiment_record
from src.theme import render_page_header, render_sidebar
from src.ui import show_error

st.set_page_config(page_title="Advanced Model Intelligence", page_icon=":material/insights:", layout="wide")
render_sidebar()
render_page_header("Advanced Model Intelligence", "Track reproducible training runs and compare model versions tied to a dataset fingerprint.")

df = st.session_state.get("df")
res = st.session_state.get("result")
fp = st.session_state.get("df_fp")
if df is None or res is None or fp is None or st.session_state.get("model_data_fp") != fp:
    st.info("Load a dataset and train a current model in Model Lab to create an experiment record.")
    st.stop()

registry = ExperimentRegistry(config.ARTIFACTS_DIR / "experiments")
st.subheader("Active model benchmark")
left, right = st.columns([1, 1])
with st.spinner("Measuring prediction time on held-out feature rows..."):
    started = time.perf_counter()
    res.pipeline.predict(res.X_test)
    elapsed = time.perf_counter() - started
metrics = evaluate(res.pipeline, res.X_test, res.y_test, res.task)
names = ("accuracy", "f1", "roc_auc") if res.task == "classification" else ("mae", "rmse", "r2")
with left:
    st.markdown(f"**{res.config.model_name}** · {res.task.title()}")
    cols = st.columns(len(names))
    for col, name in zip(cols, names):
        value = metrics.get(name)
        col.metric(name.upper(), "n/a" if value is None else f"{value:.3f}")
with right:
    st.metric("Measured prediction time", f"{elapsed * 1000 / max(1, len(res.X_test)):.3f} ms / row")
    st.caption(f"{len(res.X_test):,} held-out rows in this local run; timing includes preprocessing and prediction.")

st.subheader("Experiment history")
st.caption("The registry stores training configuration and aggregate metrics only. It never stores uploaded rows or serialized model objects.")
if st.button("Save experiment and model version", type="primary"):
    try:
        benchmark = {"predict_ms_per_row": elapsed * 1000 / max(1, len(res.X_test)), "predict_rows": len(res.X_test)}
        record = experiment_record(res, fp, metrics=metrics, benchmark=benchmark)
        registry.add(record)
        st.session_state["latest_experiment_id"] = record["experiment_id"]
        st.success(f"Saved version metadata: {record['experiment_id']}")
    except Exception as exc:  # noqa: BLE001
        show_error(exc)

try:
    records = registry.list()
except Exception as exc:  # noqa: BLE001
    show_error(exc)
    records = []
if records:
    current = [r for r in records if r.get("dataset_fingerprint") == fp]
    if current:
        st.dataframe([{
            "experiment_id": r.get("experiment_id"),
            "created_at": r.get("created_at"),
            "model": r.get("model"),
            "task": r.get("task"),
            "seed": r.get("seed"),
            "hyperparameters": r.get("hyperparameters"),
            **{f"test_{k}": v for k, v in r.get("held_out_metrics", {}).items() if isinstance(v, (int, float, str))},
        } for r in current], use_container_width=True)
    else:
        st.info("No experiment versions match the current dataset fingerprint.")
    stale = len(records) - len(current)
    if stale:
        st.caption(f"{stale} saved version(s) belong to other dataset fingerprints and are hidden from comparison.")
else:
    st.info("No experiment versions have been saved in this workspace yet.")

st.warning("Versions are metadata recipes, not serialized models. The app never loads model binaries. Recreate a version in Model Lab against the matching dataset, then verify its metrics before selecting it.")
