import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src import config
from src.data_profile import column_roles
from src.data_validation import ValidationError
from src.evaluate import METRIC_HELP, evaluate
from src.optimization import MAX_CV_FOLDS, MAX_SEARCH_BUDGET, optimize_model
from src.task_detection import infer_task
from src.train import MODELS_BY_TASK, TrainConfig, compare_models, train_model
from src.theme import record_activity, render_page_header, render_sidebar
from src.ui import require_dataset, show_error

st.set_page_config(page_title="Model Lab", page_icon=":material/model_training:", layout="wide")
render_sidebar()
render_page_header("Model Lab", "Train and compare models with leakage-aware splits and held-out evaluation.")
df = require_dataset()

target = st.selectbox("Target column", list(df.columns))
try:
    suggested = infer_task(df[target])
except ValidationError as e:
    show_error(e)
    st.stop()
task = st.radio("Task (confirm)", ["classification", "regression"], index=0 if suggested == "classification" else 1,
                horizontal=True, help=f"Suggested from the target's values: {suggested}")
roles = column_roles(df)
default_feats = [c for c in df.columns if c != target and roles[c] not in ("identifier", "constant", "empty")]
features = st.multiselect("Features", [c for c in df.columns if c != target], default=default_feats,
                          help="Identifier/constant columns are excluded by default.")
model_name = st.selectbox("Model", MODELS_BY_TASK[task])
c1, c2, c3 = st.columns(3)
test_size = c1.slider("Test fraction", *config.TEST_SIZE_BOUNDS, 0.2, 0.05)
cal_size = c2.slider("Validation/calibration fraction (of training part)", *config.CAL_SIZE_BOUNDS, 0.2, 0.05)
seed = int(c3.number_input("Random seed", 0, config.SEED_BOUNDS[1], 42))
hp = {}
with st.expander("Hyperparameters"):
    if model_name == "Logistic Regression":
        hp["C"] = st.number_input("C (inverse regularisation)", *config.HP_BOUNDS["C"], 1.0)
    if model_name in ("Decision Tree", "Random Forest"):
        hp["max_depth"] = st.slider("max_depth", *config.HP_BOUNDS["max_depth"], 6)
        hp["min_samples_leaf"] = st.slider("min_samples_leaf", *config.HP_BOUNDS["min_samples_leaf"], 1)
    if model_name == "Random Forest":
        hp["n_estimators"] = st.slider("n_estimators", *config.HP_BOUNDS["n_estimators"], 100, 10)

cfg = TrainConfig(target, task, model_name, features, test_size, cal_size, seed, hp)
b1, b2 = st.columns(2)
if b1.button("Train model", type="primary"):
    try:
        with st.spinner("Training..."):
            res = train_model(df, cfg)
        st.session_state.update(result=res, model_data_fp=st.session_state["df_fp"])
        record_activity("Model trained", f"{res.config.model_name} · {res.task}")
        for k in ("calibration", "ood", "conformal", "global_imp", "shap_explanation", "shap_explanation_identity",
                  "shap_comparison", "shap_comparison_identity",
                  "lime_explanation", "lime_explanation_identity", "optimization_result",
                  "optimization_identity", "cf_result", "cf_result_identity",
                  "model_auditor_result", "model_auditor_identity"):
            st.session_state.pop(k, None)
    except Exception as e:  # noqa: BLE001 - shown as a safe message
        show_error(e)
if b2.button("Compare all models (validation split)"):
    try:
        with st.spinner("Comparing..."):
            st.session_state["compare"] = compare_models(df, cfg)
    except Exception as e:  # noqa: BLE001
        show_error(e)
if "compare" in st.session_state:
    st.subheader("Model comparison on the validation split")
    comparison = st.session_state["compare"]
    st.dataframe(comparison.drop(columns=["confusion_matrix"], errors="ignore"), use_container_width=True)
    if task == "classification" and "confusion_matrix" in comparison:
        selected_comparison = st.selectbox("Confusion matrix for", list(comparison.index), key="comparison_cm_model")
        matrix = comparison.loc[selected_comparison, "confusion_matrix"]
        if hasattr(matrix, "shape"):
            st.dataframe(matrix, use_container_width=True)
    st.caption("Computed on the validation split with identical splits and default hyperparameters. "
               "The test set is not used, so picking a model here does not leak test labels.")

res = st.session_state.get("result")
if res is None or st.session_state.get("model_data_fp") != st.session_state.get("df_fp"):
    st.stop()

st.divider()
st.subheader("Bounded hyperparameter optimization")
if res.config.model_name == "Linear Regression":
    st.info("Linear Regression has no supported hyperparameters in this app, so there is nothing to search.")
else:
    st.caption("Search uses fit rows only with preprocessing refit inside each CV fold. Validation rows compare baseline and optimized results; test rows remain untouched.")
    opt_cols = st.columns(2)
    search_budget = opt_cols[0].slider("Search iterations", 1, MAX_SEARCH_BUDGET, 10, key="search_budget")
    cv_folds = opt_cols[1].slider("Cross-validation folds", 2, MAX_CV_FOLDS, 3, key="search_cv_folds")
    search_method = st.radio("Search strategy", ["randomized", "grid"], horizontal=True,
                             help="Both strategies are capped by the selected iteration budget and use fit rows only.")
    source_identity = (id(res), st.session_state.get("df_fp"))
    if st.session_state.get("optimization_identity") != source_identity:
        st.session_state.pop("optimization_result", None)
        st.session_state.pop("optimization_identity", None)
    if st.button("Run bounded search", key="run_optimization"):
        try:
            with st.spinner(f"Searching up to {search_budget} parameter settings across {cv_folds} folds..."):
                result = optimize_model(res, search_budget=search_budget, cv_folds=cv_folds,
                                        search_method=search_method)
            st.session_state["optimization_result"] = result
            st.session_state["optimization_identity"] = source_identity
        except Exception as e:  # noqa: BLE001
            show_error(e)

    optimization = st.session_state.get("optimization_result")
    if optimization is not None and st.session_state.get("optimization_identity") == source_identity:
        st.write("Best parameters:", optimization["best_params"])
        st.caption(f"Search strategy: {optimization['search_method']}")
        st.caption(f"CV score ({optimization['cv_scoring']}): {optimization['cv_score']:.4f} Â· "
                   f"fit {optimization['fit_seconds']:.2f}s Â· prediction {optimization['predict_ms_per_row']:.3f} ms/row Â· "
                   f"complexity {optimization['complexity']} {optimization['complexity_unit']}")
        metric_names = ("accuracy", "precision", "recall", "f1", "roc_auc") if res.task == "classification" else ("mae", "mse", "rmse", "r2")
        comparison_metrics = {
            "Baseline validation": {name: optimization["baseline_validation"].get(name) for name in metric_names},
            "Optimized validation": {name: optimization["optimized_validation"].get(name) for name in metric_names},
        }
        st.dataframe(pd.DataFrame(comparison_metrics), use_container_width=True)
        st.caption(optimization["note"])
        if st.button("Use optimized model", type="primary", key="apply_optimized_model"):
            res = optimization["result"]
            st.session_state.update(result=res, model_data_fp=st.session_state["df_fp"])
            for key in ("calibration", "ood", "conformal", "global_imp", "shap_explanation", "shap_explanation_identity",
                        "shap_comparison", "shap_comparison_identity",
                        "lime_explanation", "lime_explanation_identity", "optimization_result", "optimization_identity",
                        "cf_result", "cf_result_identity", "model_auditor_result",
                        "model_auditor_identity"):
                st.session_state.pop(key, None)
            record_activity("Optimized model selected", f"{res.config.model_name} Â· fit-only cross-validation")
            st.success("The optimized pipeline is now the active model. The held-out test results below evaluate it once.")

st.divider()
st.subheader(f"Results: {res.config.model_name} ({res.task})")
st.json({"split": res.split_info, "features": res.features, "hyperparameters": res.config.hyperparams})
for w in res.warnings:
    st.warning(w)
m = evaluate(res.pipeline, res.X_test, res.y_test, res.task)
if res.task == "classification":
    cols = st.columns(5)
    for col, k in zip(cols, ["accuracy", "precision", "recall", "f1", "roc_auc"]):
        col.metric(k, "n/a" if m[k] is None else f"{m[k]:.3f}", help=METRIC_HELP.get(k))
    st.caption(f"Precision/recall/F1 use: {m['average']}. Majority-class baseline accuracy: "
               f"{m['majority_baseline_accuracy']:.3f}.")
    st.dataframe(m["confusion_matrix"])
    st.dataframe(m["classification_report"].round(3))
    for n in m["notes"]:
        st.info(n)
else:
    cols = st.columns(4)
    for col, k in zip(cols, ["mae", "mse", "rmse", "r2"]):
        col.metric(k.upper(), "n/a" if m[k] is None else f"{m[k]:.3f}", help=METRIC_HELP.get(k))
    st.caption(f"Baseline (predict the mean) RMSE: {m['mean_baseline_rmse']:.3f}.")
    fig, ax = plt.subplots()
    ax.scatter(res.y_test, res.pipeline.predict(res.X_test), alpha=0.6)
    lo, hi = float(res.y_test.min()), float(res.y_test.max())
    ax.plot([lo, hi], [lo, hi], "k--"); ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
    st.pyplot(fig); plt.close(fig)
st.caption("All numbers above are computed on the held-out test split only.")
