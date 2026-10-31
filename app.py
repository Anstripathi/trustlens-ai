"""TrustLens AI executive overview."""
import streamlit as st

from src.evaluate import evaluate
from src.theme import PAGES, render_page_header, render_sidebar

st.set_page_config(page_title="TrustLens AI", page_icon=":material/insights:", layout="wide")
render_sidebar()
render_page_header(
    "Model intelligence, with evidence.",
    "A workspace to inspect data, evaluate models, explain predictions and assess reliability.",
    eyebrow="TRUSTLENS AI  /  EXECUTIVE OVERVIEW",
)

df = st.session_state.get("df")
res = st.session_state.get("result")
model_is_current = (
    res is not None
    and df is not None
    and st.session_state.get("model_data_fp") is not None
    and st.session_state.get("model_data_fp") == st.session_state.get("df_fp")
)

st.subheader("Workspace status")
status_cols = st.columns(3)
with status_cols[0].container(border=True, height="stretch"):
    st.caption("DATASET")
    if df is None:
        st.markdown("**No dataset loaded**")
        st.caption("Upload a CSV or open a demo dataset to begin.")
    else:
        st.markdown(f"**{st.session_state.get('dataset_name', 'Loaded dataset')}**")
        st.caption(f"{df.shape[0]:,} rows · {df.shape[1]:,} columns")
with status_cols[1].container(border=True, height="stretch"):
    st.caption("MODEL")
    if model_is_current:
        st.markdown(f"**{res.config.model_name}**")
        st.caption(f"{res.task.title()} · trained on the current dataset")
    elif res is not None:
        st.markdown("**Retraining required**")
        st.caption("The dataset changed after this model was trained.")
    else:
        st.markdown("**No trained model**")
        st.caption("Train a model in Model Lab to see held-out results.")
with status_cols[2].container(border=True, height="stretch"):
    st.caption("RELIABILITY")
    calibration = st.session_state.get("calibration")
    conformal = st.session_state.get("conformal")
    ood = st.session_state.get("ood")
    if calibration is not None:
        st.markdown("**Calibration assessed**")
        st.caption(
            f"Calibrated Brier {calibration['calibrated']['brier']:.3f} · "
            f"ECE {calibration['calibrated']['ece']:.3f}"
        )
    elif conformal is not None:
        st.markdown("**Prediction intervals assessed**")
        st.caption(f"Test split coverage {conformal['empirical_test_coverage']:.1%}")
    elif ood is not None:
        flagged_rate = ood["flagged_rows"] / ood["total_rows"] if ood["total_rows"] else 0.0
        st.markdown("**Outlier check completed**")
        st.caption(f"Test rows flagged {flagged_rate:.1%}")
    else:
        st.markdown("**Not assessed yet**")
        st.caption("Reliability indicators appear after an assessment is run.")

if model_is_current:
    st.subheader("Held-out model performance")
    metrics = evaluate(res.pipeline, res.X_test, res.y_test, res.task)
    metric_names = (
        ("accuracy", "f1", "roc_auc") if res.task == "classification"
        else ("mae", "rmse", "r2")
    )
    metric_cols = st.columns(len(metric_names))
    for col, name in zip(metric_cols, metric_names):
        value = metrics.get(name)
        col.metric(name.upper(), "n/a" if value is None else f"{value:.3f}", border=True)
    st.caption("Computed on the held-out test split; these results describe this dataset and split only.")
elif res is not None and df is not None:
    st.warning("This model belongs to a previous dataset. Retrain it in Model Lab before using its results.")

left, right = st.columns([1.25, 1], gap="large")
with left:
    st.subheader("Explore the workspace")
    st.markdown("Review dataset quality, train and compare models, or inspect model behavior and reliability.")
    for title, path, icon in PAGES:
        st.page_link(path, label=f"Open {title}", icon=f":material/{icon}:")
with right:
    st.subheader("Recent activity")
    activity = st.session_state.get("activity", [])
    if not activity:
        st.info("Actions performed in this session will appear here.")
    else:
        for event in activity[:5]:
            with st.container(border=True):
                st.markdown(f"**{event['action']}** · {event['detail']}")
                st.caption(event["at"].strftime("%b %d, %Y · %H:%M %Z"))

st.divider()
st.caption(
    "TrustLens AI does not certify a model as trustworthy, unbiased, causal or production-ready. "
    "Metrics describe behavior on your data only. Uploaded data is processed in memory for your session and is not saved by the app."
)
