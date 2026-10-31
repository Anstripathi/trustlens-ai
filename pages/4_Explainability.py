import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.explain import CAUSAL_NOTE, global_importance, local_explanation
from src.counterfactual import CounterfactualError, generate_counterfactuals
from src.lime_explain import LimeUnavailableError, explain_lime
from src.shap_explain import ShapUnavailableError, compare_shap_importance, explain_shap
from src.theme import render_page_header, render_sidebar
from src.ui import require_dataset, require_model, show_error

st.set_page_config(page_title="Explainability", page_icon=":material/query_stats:", layout="wide")
render_sidebar()
render_page_header("Explainability", "Inspect global feature effects and individual model outputs.")
require_dataset()
res = require_model()

tabs = st.tabs(["SHAP", "LIME", "Counterfactual", "Permutation & baseline methods"])

with tabs[0]:
    st.markdown("### SHAP explanations")
    st.caption(
        "Permutation SHAP estimates each original feature's contribution to a selected model output. "
        "Results explain model behavior; they do not describe causes."
    )
    explain_mode = st.radio(
        "Explanation view", ["Global summary", "Single prediction"], horizontal=True,
        key="shap_explain_mode",
    )
    kwargs = {}
    if res.task == "classification":
        selected_class = st.selectbox(
            "Class probability to explain", list(res.classes), index=len(res.classes) - 1,
            key="shap_class_label",
        )
        kwargs["class_label"] = selected_class

    if explain_mode == "Global summary":
        n_rows = st.slider("Test rows to sample", 1, 25, min(10, len(res.X_test)),
                           help="A deterministic sample from the held-out test split; capped to bound compute.")
        rows = res.X_test.sample(min(n_rows, len(res.X_test)), random_state=res.config.seed).sort_index()
    else:
        row_index = st.number_input("Test row position", min_value=0, max_value=len(res.X_test) - 1,
                                    value=0, step=1, key="shap_row_position")
        rows = res.X_test.iloc[[int(row_index)]]

    identity = (id(res), st.session_state.get("df_fp"), explain_mode,
                tuple(rows.index.tolist()), kwargs.get("class_label"))
    if st.session_state.get("shap_comparison_identity") != identity:
        st.session_state.pop("shap_comparison", None)
        st.session_state.pop("shap_comparison_identity", None)
    if st.session_state.get("shap_explanation_identity") != identity:
        st.session_state.pop("shap_explanation", None)
        st.session_state.pop("shap_explanation_identity", None)

    if st.button("Compute SHAP explanation", type="primary", key="compute_shap"):
        try:
            with st.spinner("Computing bounded permutation SHAP values..."):
                explanation = explain_shap(res, rows, **kwargs)
            st.session_state["shap_explanation"] = explanation
            st.session_state["shap_explanation_identity"] = identity
        except ShapUnavailableError as exc:
            st.warning(str(exc))
        except Exception as exc:  # noqa: BLE001
            show_error(exc)

    explanation = st.session_state.get("shap_explanation")
    if explanation is not None and st.session_state.get("shap_explanation_identity") == identity:
        st.caption(
            f"Explained {explanation['explained_rows']} row(s) with {explanation['background_rows']} "
            f"fit-split background rows. Output: {explanation['output_name']}."
        )
        if explain_mode == "Global summary":
            importance = explanation["importance"].head(15).sort_values("mean_abs_shap")
            left, right = st.columns([1, 1.4], gap="large")
            with left:
                st.markdown("**Mean absolute contribution**")
                fig, ax = plt.subplots(figsize=(7, max(3, len(importance) * 0.32)))
                ax.barh(importance.feature, importance.mean_abs_shap, color="#6488ff")
                ax.set_xlabel(f"Mean |SHAP value| for {explanation['output_name']}")
                ax.grid(axis="x", alpha=0.18)
                st.pyplot(fig, clear_figure=True)
                plt.close(fig)
            with right:
                st.markdown("**Contribution distribution**")
                vals = explanation["values"]
                names = explanation["rows"].columns.tolist()
                order = explanation["importance"].feature.tolist()[:15][::-1]
                positions = {name: idx for idx, name in enumerate(names)}
                fig, ax = plt.subplots(figsize=(8, max(3, len(order) * 0.32)))
                for y_pos, name in enumerate(order):
                    x = vals[:, positions[name]]
                    jitter = (np.arange(len(x)) % 7 - 3) * 0.035
                    ax.scatter(x, y_pos + jitter, s=20, alpha=0.72, color="#8b78e6")
                ax.axvline(0, color="#aeb8ca", linewidth=0.8)
                ax.set_yticks(range(len(order)), order)
                ax.set_xlabel("SHAP contribution (positive raises the selected output)")
                ax.grid(axis="x", alpha=0.18)
                st.pyplot(fig, clear_figure=True)
                plt.close(fig)
            st.dataframe(explanation["importance"], use_container_width=True, hide_index=True)
            if st.button("Compare SHAP importance across model families", key="compare_shap_models"):
                try:
                    with st.spinner("Fitting compatible model families on the same fit split..."):
                        comparison = compare_shap_importance(
                            res, rows, class_label=kwargs.get("class_label"),
                        )
                    st.session_state["shap_comparison"] = comparison
                    st.session_state["shap_comparison_identity"] = identity
                except Exception as exc:  # noqa: BLE001
                    show_error(exc)
            comparison = st.session_state.get("shap_comparison")
            if comparison is not None and st.session_state.get("shap_comparison_identity") == identity:
                st.markdown("**Mean absolute SHAP by model family**")
                table = comparison["importance"].loc[
                    explanation["importance"].feature.head(15).tolist()
                ]
                table = table.loc[table.mean(axis=1).sort_values().index]
                fig, ax = plt.subplots(figsize=(9, max(3, len(table) * 0.35)))
                table.plot(kind="barh", ax=ax)
                ax.set_xlabel(f"Mean |SHAP value| for {explanation['output_name']}")
                st.pyplot(fig, clear_figure=True)
                plt.close(fig)
                st.dataframe(table, use_container_width=True)
                st.caption(comparison["note"])
                for model_name, reason in comparison["failures"].items():
                    st.caption(f"{model_name} unavailable: {reason}")
        else:
            import shap

            baseline = float(explanation["base_values"][0])
            prediction = float(explanation["predictions"][0])
            m1, m2, m3 = st.columns(3)
            m1.metric("Baseline output", f"{baseline:.4f}")
            m2.metric("Explained output", f"{prediction:.4f}")
            m3.metric("Selected class" if res.task == "classification" else "Output", explanation["output_name"])
            shap_row = shap.Explanation(
                values=explanation["values"][0], base_values=baseline,
                feature_names=explanation["rows"].columns.tolist(),
            )
            fig = plt.figure(figsize=(10, max(4, len(res.features) * 0.35)))
            shap.plots.waterfall(shap_row, max_display=15, show=False)
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)
            contributions = explanation["contributions"].iloc[0].rename("SHAP contribution").to_frame()
            contributions.insert(0, "Feature value", explanation["rows"].iloc[0])
            st.dataframe(contributions.reindex(
                contributions["SHAP contribution"].abs().sort_values(ascending=False).index
            ), use_container_width=True)
        st.caption(explanation["note"])
        st.caption("The displayed model output is reconstructed from the sampled background and SHAP contributions.")

with tabs[1]:
    st.markdown("### LIME local approximation")
    st.caption(
        "LIME perturbs this observation and fits a local surrogate. Its weights approximate nearby model behavior; "
        "they are not global feature importance or causal effects."
    )
    lime_row_index = st.number_input("Test row position", min_value=0, max_value=len(res.X_test) - 1,
                                     value=0, step=1, key="lime_row_position")
    lime_row = res.X_test.iloc[[int(lime_row_index)]]
    lime_kwargs = {}
    if res.task == "classification":
        lime_class = st.selectbox("Class probability to explain", list(res.classes),
                                  index=len(res.classes) - 1, key="lime_class_label")
        lime_kwargs["class_label"] = lime_class
    lime_samples = st.slider("Perturbed samples", 100, 5_000, 1_000, 100,
                             help="More samples can stabilize the local fit but increase runtime.")
    feature_limit = min(20, len(res.features))
    lime_features = st.slider("Features shown", 1, feature_limit, min(10, feature_limit),
                              key="lime_num_features")
    lime_identity = (id(res), st.session_state.get("df_fp"), int(lime_row_index),
                     lime_kwargs.get("class_label"), lime_samples, lime_features)
    if st.session_state.get("lime_explanation_identity") != lime_identity:
        st.session_state.pop("lime_explanation", None)
        st.session_state.pop("lime_explanation_identity", None)
    if st.button("Compute LIME explanation", type="primary", key="compute_lime"):
        try:
            with st.spinner("Sampling nearby inputs and fitting the local surrogate..."):
                lime_result = explain_lime(
                    res, lime_row, num_samples=lime_samples,
                    num_features=lime_features, **lime_kwargs,
                )
            st.session_state["lime_explanation"] = lime_result
            st.session_state["lime_explanation_identity"] = lime_identity
        except LimeUnavailableError as exc:
            st.warning(str(exc))
        except Exception as exc:  # noqa: BLE001
            show_error(exc)

    lime_result = st.session_state.get("lime_explanation")
    if lime_result is not None and st.session_state.get("lime_explanation_identity") == lime_identity:
        c1, c2, c3 = st.columns(3)
        c1.metric("Model output", f"{lime_result['model_output']:.4f}")
        c2.metric("Local surrogate output", f"{lime_result['local_prediction']:.4f}")
        c3.metric("Local fit R²", f"{lime_result['local_r2']:.3f}")
        st.caption(f"Explained output: {lime_result['output_name']}. {lime_result['background_rows']} fit rows used for the local feature distribution.")
        contribution_table = lime_result["contributions"].sort_values("weight")
        fig, ax = plt.subplots(figsize=(9, max(3, len(contribution_table) * 0.4)))
        ax.barh(contribution_table.feature, contribution_table.weight,
                color=["#ef6461" if value < 0 else "#2ebd85" for value in contribution_table.weight])
        ax.axvline(0, color="#aeb8ca", linewidth=0.8)
        ax.set_xlabel("Local surrogate weight (sign is relative to selected output)")
        ax.grid(axis="x", alpha=0.18)
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)
        st.dataframe(lime_result["contributions"], use_container_width=True, hide_index=True)
        st.caption(lime_result["note"])

        shap_result = st.session_state.get("shap_explanation")
        shap_identity = st.session_state.get("shap_explanation_identity")
        same_shap_target = (
            shap_result is not None
            and isinstance(shap_identity, tuple)
            and len(shap_identity) == 5
            and shap_identity[0] == id(res)
            and shap_identity[1] == st.session_state.get("df_fp")
            and shap_identity[2] == "Single prediction"
            and shap_identity[3] == (lime_row.index[0],)
            and shap_identity[4] == lime_kwargs.get("class_label")
        )
        if same_shap_target:
            shap_weights = shap_result["contributions"].iloc[0].rename("SHAP value")
            lime_weights = lime_result["contributions"].set_index("feature")["weight"].rename("LIME weight")
            comparison = shap_weights.to_frame().join(lime_weights, how="inner")
            sign_match = (np.sign(comparison["SHAP value"]) == np.sign(comparison["LIME weight"])).mean()
            st.markdown("**LIME and SHAP direction comparison**")
            st.metric("Features with matching contribution direction", f"{sign_match:.0%}")
            st.caption("Weights have different definitions and should not be compared as equal magnitudes.")
            st.dataframe(comparison, use_container_width=True)
        else:
            st.info("Compute a single-prediction SHAP explanation for this same row and class to compare contribution directions.")

with tabs[2]:
    st.markdown("### Counterfactual scenarios")
    st.caption(
        "Search for a nearby model-input scenario that meets a selected prediction goal. "
        "A counterfactual is not a causal intervention or a guarantee about real-world outcomes."
    )
    cf_position = st.number_input("Test row position", min_value=0, max_value=len(res.X_test) - 1,
                                  value=0, step=1, key="cf_row_position")
    cf_row = res.X_test.iloc[[int(cf_position)]]
    cf_options = {}
    if res.task == "classification":
        cf_target = st.selectbox("Desired predicted class", list(res.classes), key="cf_target_class")
        cf_options["desired_class"] = cf_target
    else:
        y_min, y_max = float(res.y_fit.min()), float(res.y_fit.max())
        if y_min == y_max:
            st.warning("The training target has no range to search for.")
        margin = max((y_max - y_min) * 0.08, 1e-6)
        default_low = max(y_min, float(res.y_fit.quantile(0.7)) - margin)
        default_high = min(y_max, float(res.y_fit.quantile(0.7)) + margin)
        cf_low = st.number_input("Desired prediction minimum", value=default_low, key="cf_target_low")
        cf_high = st.number_input("Desired prediction maximum", value=default_high, key="cf_target_high")
        cf_options["target_range"] = (cf_low, cf_high)

    immutable_features = st.multiselect("Immutable features", res.features, key="cf_immutable")
    numeric_features = [name for name in res.features if res.feature_kinds[name] == "numeric"]
    feature_ranges = {}
    if numeric_features:
        constrained_feature = st.selectbox(
            "Numeric feature range constraint", ["No additional range"] + numeric_features,
            key="cf_range_feature",
        )
        if constrained_feature != "No additional range":
            series = res.X_fit[constrained_feature].dropna().astype(float)
            lower, upper = float(series.min()), float(series.max())
            if lower < upper:
                bounds = st.slider(f"Allowed values for {constrained_feature}", lower, upper,
                                   (lower, upper), key="cf_range_bounds")
                feature_ranges[constrained_feature] = bounds
            else:
                st.caption(f"{constrained_feature} has no training range to constrain.")
    cf_budget = st.slider("Candidate evaluation budget", 10, 200, 100, 10, key="cf_candidate_budget")
    cf_changes = st.slider("Maximum changed features", 1, 3, 2, key="cf_max_changes")
    cf_identity = (id(res), st.session_state.get("df_fp"), int(cf_position),
                   tuple(sorted(immutable_features)), tuple(sorted(feature_ranges.items())),
                   tuple(cf_options.values()), cf_budget, cf_changes)
    if st.session_state.get("cf_result_identity") != cf_identity:
        st.session_state.pop("cf_result", None)
        st.session_state.pop("cf_result_identity", None)
    if st.button("Search counterfactual scenarios", type="primary", key="compute_counterfactual"):
        try:
            with st.spinner("Searching bounded candidate scenarios..."):
                cf_result = generate_counterfactuals(
                    res, cf_row, immutable_features=immutable_features,
                    feature_ranges=feature_ranges, max_candidates=cf_budget,
                    max_changes=cf_changes, **cf_options,
                )
            st.session_state["cf_result"] = cf_result
            st.session_state["cf_result_identity"] = cf_identity
        except CounterfactualError as exc:
            st.warning(str(exc))
        except Exception as exc:  # noqa: BLE001
            show_error(exc)

    cf_result = st.session_state.get("cf_result")
    if cf_result is not None and st.session_state.get("cf_result_identity") == cf_identity:
        if cf_result.get("already_satisfied"):
            st.success(cf_result["reason"])
            st.metric("Original model prediction", str(cf_result["original_prediction"]))
        elif not cf_result["found"]:
            st.warning(cf_result["reason"])
            st.caption(f"Evaluated {cf_result['evaluated']} candidate(s) under the configured constraints.")
        else:
            best = cf_result["candidates"].iloc[0]
            left, right = st.columns(2)
            left.metric("Original prediction", str(cf_result["original_prediction"]))
            right.metric("Counterfactual prediction", str(best["prediction"]))
            if res.task == "classification":
                st.caption(f"Estimated raw model probability for {cf_target!r}: {best['desired_output']:.3f}.")
            st.caption(f"{cf_result['reason']} Evaluated {cf_result['evaluated']} candidate(s).")
            changed = set(best["changed_features"])
            changes = pd.DataFrame([
                {"Feature": name, "Original value": cf_row.iloc[0][name],
                 "Candidate value": best[name], "Changed": name in changed}
                for name in res.features
            ])
            st.dataframe(changes[changes.Changed], use_container_width=True, hide_index=True)
            with st.expander("Other valid candidates"):
                st.dataframe(cf_result["candidates"].head(10), use_container_width=True, hide_index=True)
            st.info(cf_result["note"])

with tabs[3]:
    st.info(CAUSAL_NOTE + " These methods are retained alongside SHAP for comparison.")
    st.subheader("Global: permutation importance")
    if st.button("Compute global importance", key="compute_permutation"):
        try:
            with st.spinner("Permuting features on the test split..."):
                g = global_importance(res)
            if not g["available"]:
                st.warning(g["reason"])
            else:
                st.session_state["global_imp"] = g
        except Exception as e:  # noqa: BLE001
            show_error(e)
    g = st.session_state.get("global_imp")
    if g is not None:
        t = g["table"].head(15)[::-1]
        fig, ax = plt.subplots()
        ax.barh(t.feature, t.importance_mean, xerr=t.importance_std)
        ax.set_xlabel(f"Score drop ({g['scoring']})")
        st.pyplot(fig); plt.close(fig)
        st.caption(g["method"])

    st.subheader("Local: baseline substitution")
    i = st.number_input("Test row position", 0, len(res.X_test) - 1, 0, key="baseline_row_position")
    row = res.X_test.iloc[[int(i)]]
    try:
        ex = local_explanation(res, row)
        st.write(f"Model prediction: **{ex['prediction']}** (actual: {res.y_test.iloc[int(i)]})")
        st.write(f"Explained quantity: {ex['output_explained']} = {ex['model_output']:.3f}")
        t = ex["contributions"].head(10)[::-1]
        fig, ax = plt.subplots()
        ax.barh(t.feature, t.contribution, color=["tab:green" if v > 0 else "tab:red" for v in t.contribution])
        st.pyplot(fig); plt.close(fig)
        st.dataframe(ex["contributions"], use_container_width=True)
        st.caption(ex["method"])
    except Exception as e:  # noqa: BLE001
        show_error(e)
