import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src import config
from src.artifact_manager import reject_model_artifact
from src.calibration import assess_calibration, confidence_flags
from src.data_validation import ValidationError, load_csv_bytes
from src.drift import drift_report
from src.theme import render_page_header, render_sidebar
from src.ui import require_model, show_error
from src.uncertainty import OODDetector, conformal_intervals, fit_ood

st.set_page_config(page_title="Reliability Monitor", page_icon=":material/verified_user:", layout="wide")
render_sidebar()
render_page_header("Reliability Monitor", "Assess confidence, calibration, unusual inputs and dataset drift.")
tab_pred, tab_cal, tab_ood, tab_drift = st.tabs(["Confidence / intervals", "Calibration", "Out-of-distribution", "Data drift"])

with tab_drift:
    st.write("Compare a reference dataset with a current dataset (no trained model needed).")
    ref_f = st.file_uploader("Reference CSV", key="ref", type=None)
    cur_f = st.file_uploader("Current CSV", key="cur", type=None)
    alpha = st.select_slider("Significance level α", [0.01, 0.05, 0.1], 0.05)
    corr = st.radio("Multiple-comparison correction", ["bh", "bonferroni"], horizontal=True,
                    format_func=lambda x: "Benjamini-Hochberg (FDR)" if x == "bh" else "Bonferroni")
    if ref_f is not None and cur_f is not None:
        try:
            for f in (ref_f, cur_f):
                reject_model_artifact(f.name)
            ref = load_csv_bytes(ref_f.getvalue(), ref_f.name).clean
            cur = load_csv_bytes(cur_f.getvalue(), cur_f.name).clean
            rep = drift_report(ref, cur, alpha=alpha, correction=corr)
            s = rep["schema"]
            if s["missing_in_current"] or s["new_in_current"] or s["type_mismatch"]:
                st.warning(f"Schema differences - missing in current: {s['missing_in_current']}, "
                           f"new in current: {s['new_in_current']}, type mismatch: {s['type_mismatch']}")
            if not s["common"]:
                st.error("The datasets share no columns.")
            else:
                st.write(rep["summary"])
                st.dataframe(rep["table"], use_container_width=True)
            st.info(rep["disclaimer"])
            st.caption("'drift' = significant after correction AND effect size above the practical threshold; "
                       "with large samples tiny differences become significant, so both are required.")
        except ValidationError as e:
            show_error(e)
    else:
        st.info("Upload both files to run the comparison.")

res = st.session_state.get("result")
with tab_pred:
    if res is None:
        st.info("Train a model in the Model Lab to use this tab.")
    else:
        require_model()
        i = int(st.number_input("Test row position", 0, len(res.X_test) - 1, 0, key="ri"))
        row = res.X_test.iloc[[i]]
        if res.task == "classification":
            thr = st.slider("Low-confidence threshold", 0.3, 0.99, config.DEFAULT_CONFIDENCE_THRESHOLD, 0.01)
            raw_values = res.pipeline.predict_proba(row)[0]
            proba = pd.Series(raw_values, index=map(str, res.classes))
            st.write("Predicted:", res.pipeline.predict(row)[0])
            st.caption("Raw model class probabilities")
            st.bar_chart(proba)
            top_probability = float(np.max(raw_values))
            entropy = float(-np.sum(raw_values * np.log(np.clip(raw_values, 1e-12, 1.0))))
            normalized_entropy = entropy / np.log(len(raw_values)) if len(raw_values) > 1 else 0.0
            c1, c2 = st.columns(2)
            c1.metric("Top raw probability", f"{top_probability:.3f}",
                      help="Uncalibrated model output; not a guarantee that this prediction is correct.")
            c2.metric("Prediction entropy", f"{normalized_entropy:.3f}",
                      help="Normalized distributional ambiguity (0–1). It does not measure model or data uncertainty.")
            if confidence_flags([top_probability], thr)[0]:
                st.warning(f"Low top probability: {top_probability:.2f} < {thr:.2f}.")
            calibration = st.session_state.get("calibration")
            if calibration is not None and calibration.get("available"):
                if len(res.classes) == 2:
                    positive = float(calibration["calibrator"].transform([raw_values[1]])[0])
                    calibrated_values = np.array([1 - positive, positive])
                else:
                    calibrated_values = calibration["calibrator"].transform(raw_values.reshape(1, -1))[0]
                st.caption("Calibrated probabilities (class order shown above)")
                st.bar_chart(pd.Series(calibrated_values, index=map(str, res.classes)))
            st.caption("Calibration is assessed against held-out frequencies below. Prediction entropy is distributional ambiguity; it is not epistemic/model uncertainty or data uncertainty.")
        else:
            alpha = st.select_slider("Interval miscoverage α", [0.05, 0.1, 0.2], config.DEFAULT_ALPHA)
            ci = conformal_intervals(res, row, alpha)
            st.session_state["conformal"] = ci if ci["available"] else None
            st.write("Point prediction:", float(res.pipeline.predict(row)[0]))
            if ci["available"]:
                r = ci["intervals"].iloc[0]
                st.success(f"{1 - alpha:.0%} split-conformal interval: [{r.lower:.3f}, {r.upper:.3f}]")
                st.caption(f"Empirical coverage on the test split: {ci['empirical_test_coverage']:.2%}. " + " ".join(ci["notes"]))
            else:
                st.warning("No prediction interval: " + ci["reason"])

with tab_cal:
    if res is None:
        st.info("Train a model in the Model Lab to use this tab.")
    elif res.task != "classification":
        st.info("Calibration applies to classification.")
    else:
        method = st.radio("Calibrator", ["sigmoid", "isotonic"], horizontal=True)
        a = assess_calibration(res, method)
        if not a["available"]:
            st.warning(a["reason"])
        else:
            c1, c2 = st.columns(2)
            c1.metric("Brier (raw → calibrated)", f"{a['raw']['brier']:.4f} → {a['calibrated']['brier']:.4f}")
            c2.metric("ECE (raw → calibrated)", f"{a['raw']['ece']:.4f} → {a['calibrated']['ece']:.4f}")
            st.session_state["calibration"] = a
            fig, ax = plt.subplots()
            ax.plot([0, 1], [0, 1], "k--")
            for key, lab in (("reliability_raw", "raw"), ("reliability_calibrated", "calibrated")):
                t = a[key]; ax.plot(t.mean_predicted, t.observed_rate, "o-", label=lab)
            ax.set_xlabel("Mean predicted probability / top-label confidence")
            ax.set_ylabel("Observed positive rate / top-label accuracy")
            ax.legend()
            st.pyplot(fig); plt.close(fig)
            st.caption("Binary calibration tracks the selected class probability. Multiclass calibration uses top-label confidence and accuracy; Brier is the multiclass squared-probability score.")
            for n in a["notes"]:
                st.caption("• " + n)

with tab_ood:
    if res is None:
        st.info("Train a model in the Model Lab to use this tab.")
    else:
        rate = st.slider("Target flag rate on typical data", 0.01, 0.2, config.DEFAULT_OOD_FLAG_RATE, 0.01)
        try:
            det = fit_ood(res, rate)
            sc = det.score(res.X_test)
            st.session_state["ood"] = {
                "flagged_rows": int(sc.is_out_of_distribution.sum()),
                "total_rows": len(sc),
            }
            st.metric("Test rows flagged", f"{sc.is_out_of_distribution.mean():.1%}")
            i = int(st.number_input("Test row position", 0, len(res.X_test) - 1, 0, key="oi"))
            one = sc.iloc[i]
            (st.warning if one.is_out_of_distribution else st.success)(
                f"Row {i}: {'flagged as unusual' if one.is_out_of_distribution else 'not flagged'} (score {one.ood_score:.3f}).")
            st.caption(OODDetector.notes)
        except Exception as e:  # noqa: BLE001
            show_error(e)
