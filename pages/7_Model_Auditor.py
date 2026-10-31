"""Unified, evidence-backed audit over the current TrustLens session."""
from __future__ import annotations

import hashlib
import json

import pandas as pd
import streamlit as st

from src.auditor import AuditRequest, run_audit
from src.auditor.reporting import csv_report_bytes, json_report_bytes
from src.artifact_manager import dataset_fingerprint
from src.data_validation import ValidationError, load_csv_bytes
from src.security.schemas import MAX_SECURITY_INPUT_CHARS
from src.theme import record_activity, render_page_header, render_sidebar

st.set_page_config(page_title="Model Auditor", page_icon=":material/policy:", layout="wide")
render_sidebar()
render_page_header(
    "Autonomous AI Model Auditor",
    "Coordinate evidence-backed dataset, model, explanation, reliability and local security checks.",
)
st.info(
    "Audit results are screening evidence for human review. They do not certify safety, fairness, "
    "trustworthiness or production readiness. No target LLM is called or penetration-tested."
)

dataset = st.session_state.get("df")
if not isinstance(dataset, pd.DataFrame):
    dataset = None
dataset_name = st.session_state.get("dataset_name", "Current session dataset")
dataset_fp = st.session_state.get("df_fp")
active_model = st.session_state.get("result")
model_fp = st.session_state.get("model_data_fp")
model_is_current = active_model is not None and dataset is not None and model_fp == dataset_fp

status_cols = st.columns(3)
with status_cols[0].container(border=True):
    st.caption("DATASET")
    if dataset is None:
        st.markdown("**No dataset loaded**")
        st.caption("Load a dataset in Data Explorer before starting an audit.")
    else:
        st.markdown(f"**{dataset_name}**")
        st.caption(f"{len(dataset):,} rows · {dataset.shape[1]} columns")
with status_cols[1].container(border=True):
    st.caption("ACTIVE MODEL")
    if model_is_current:
        st.markdown(f"**{active_model.config.model_name}**")
        st.caption(f"{active_model.task.title()} · current dataset")
    elif active_model is not None:
        st.markdown("**Model is stale**")
        st.caption("It belongs to a different dataset and will not be audited as current.")
    else:
        st.markdown("**No trained model**")
        st.caption("Dataset checks can run; model-dependent checks will be marked not evaluated.")
with status_cols[2].container(border=True):
    st.caption("LATEST AUDIT")
    latest = st.session_state.get("model_auditor_result")
    if latest is None or latest.risk is None:
        st.markdown("**Not run in this session**")
        st.caption("The auditor does not display default or fabricated metrics.")
    else:
        st.markdown(f"**{latest.risk.risk_level}**")
        st.caption(f"{latest.risk.evaluated_checks} checks evaluated")

if active_model is not None and not model_is_current:
    st.warning("The stored model fingerprint does not match the loaded dataset. Model-dependent checks will not run.")

st.subheader("Assessment configuration")
st.caption("The primary dataset and model come from the current Streamlit session. Upload reference/current files only if you want drift analysis.")
left_upload, right_upload = st.columns(2)
with left_upload:
    reference_upload = st.file_uploader("Optional drift reference CSV", type=["csv"], key="auditor_reference_csv")
with right_upload:
    current_upload = st.file_uploader("Optional drift current CSV", type=["csv"], key="auditor_current_csv")

reference_data = current_data = None
for upload, label, assign in ((reference_upload, "reference", "reference"),
                              (current_upload, "current", "current")):
    if upload is None:
        continue
    try:
        loaded = load_csv_bytes(upload.getvalue(), upload.name)
        if assign == "reference":
            reference_data = loaded.clean
        else:
            current_data = loaded.clean
    except ValidationError as exc:
        st.error(f"{label.title()} dataset: {exc}")
if (reference_upload is None) != (current_upload is None):
    st.caption("Drift will remain not evaluated until both CSVs are provided.")

with st.form("model_audit_configuration"):
    if dataset is not None and len(dataset.columns):
        target_options = ["(none)", *map(str, dataset.columns)]
        default_target = getattr(getattr(active_model, "config", None), "target", "(none)")
        target_index = target_options.index(default_target) if default_target in target_options else 0
        selected_target = st.selectbox("Target column for data audit", target_options, index=target_index)
        target_column = None if selected_target == "(none)" else selected_target
    else:
        target_column = None
        st.selectbox("Target column for data audit", ["No dataset loaded"], disabled=True)

    include_explanations = st.checkbox("Run bounded explainability checks", value=True,
                                       disabled=not model_is_current)
    include_reliability = st.checkbox("Run reliability and uncertainty checks", value=True,
                                      disabled=not model_is_current)
    benchmark_models = st.checkbox(
        "Explicitly benchmark supported models on validation data",
        value=False, disabled=not model_is_current,
        help="Fits bounded candidate models for comparison; the test split is not used for selection and the active model is not replaced.",
    )
    run_cross_validation = st.checkbox(
        "Run optional 3-fold cross-validation on fit rows",
        value=False, disabled=not model_is_current,
        help="Fits clones of the selected pipeline on at most 5,000 fit rows; it does not modify the active model or use test rows.",
    )
    include_security = st.checkbox("Run local security screening", value=False)
    with st.expander("Security screening options", expanded=False):
        security_checks = st.multiselect(
            "Checks", ["prompt_injection", "jailbreak", "pii", "toxicity"],
            default=["prompt_injection", "jailbreak", "pii", "toxicity"],
            format_func=lambda value: {
                "prompt_injection": "Prompt injection patterns",
                "jailbreak": "Built-in jailbreak scanner fixtures (not a target-model test)",
                "pii": "PII and sensitive-data patterns",
                "toxicity": "Unsafe-output lexical patterns",
            }[value],
        )
        prompt_sample = st.text_area(
            "Optional prompt sample", max_chars=MAX_SECURITY_INPUT_CHARS, height=100,
            help="Processed locally; omitted from assessment results and reports.",
        )
        output_sample = st.text_area(
            "Optional model-output sample", max_chars=MAX_SECURITY_INPUT_CHARS, height=100,
            help="Processed locally; omitted from assessment results and reports.",
        )
        scan_dataset_columns = st.checkbox(
            "Explicitly scan selected columns from the active dataset for sensitive patterns",
            value=False, disabled=dataset is None,
        )
        if dataset is not None:
            selected_scan_columns = st.multiselect(
                "Dataset columns to scan", list(dataset.columns), default=[],
                disabled=not scan_dataset_columns,
            )
        else:
            selected_scan_columns = []
    submitted = st.form_submit_button("Run model audit", type="primary", disabled=dataset is None)

prompt_hash = hashlib.sha256((prompt_sample + "\0" + output_sample).encode("utf-8")).hexdigest()
reference_fp = dataset_fingerprint(reference_data) if reference_data is not None else None
current_fp = dataset_fingerprint(current_data) if current_data is not None else None
audit_identity = (
    dataset_fp, id(active_model) if model_is_current else None, target_column,
    include_explanations, include_reliability, benchmark_models, run_cross_validation, include_security,
    tuple(sorted(security_checks)), tuple(selected_scan_columns) if scan_dataset_columns else (),
    prompt_hash, reference_fp, current_fp,
)
stored_identity = st.session_state.get("model_auditor_identity")
if stored_identity is not None and stored_identity != audit_identity:
    st.session_state.pop("model_auditor_result", None)
    st.session_state.pop("model_auditor_identity", None)

if submitted and dataset is not None:
    request = AuditRequest(
        dataset=dataset,
        dataset_name=dataset_name,
        dataset_fingerprint=dataset_fp,
        model=active_model,
        model_fingerprint=model_fp,
        reference_dataset=reference_data,
        current_dataset=current_data,
        target_column=target_column,
        benchmark_models=benchmark_models,
        run_cross_validation=run_cross_validation,
        run_explainability=include_explanations,
        run_reliability=include_reliability,
        run_security=include_security,
        security_checks=tuple(security_checks),
        prompt_sample=prompt_sample,
        output_sample=output_sample,
        dataset_scan_columns=tuple(selected_scan_columns) if scan_dataset_columns else (),
        execution_limits={"security_dataset_cells": 500},
    )
    try:
        with st.spinner("Running bounded audit stages. Successful results are retained if a stage fails..."):
            audited = run_audit(request)
        st.session_state["model_auditor_result"] = audited
        st.session_state["model_auditor_identity"] = audit_identity
        record_activity("Model audit completed", f"{audited.risk.risk_level} · {audited.risk.evaluated_checks} check(s)")
    except Exception:  # noqa: BLE001 - do not expose raw data or input in exceptions
        st.error("The audit could not be initialized. Check the selected dataset and options, then try again.")

result = st.session_state.get("model_auditor_result")
if result is None:
    st.subheader("Audit results")
    st.info("Configure the audit and select **Run model audit**. Checks without required evidence will be marked not evaluated.")
    st.stop()

st.divider()
st.subheader("Executive assessment")
risk = result.risk
summary_cols = st.columns(4)
summary_cols[0].metric("Overall risk", risk.risk_level)
summary_cols[1].metric("Risk score", "Withheld" if risk.score is None else f"{risk.score:.1f} / 100")
summary_cols[2].metric("Checks evaluated", f"{risk.evaluated_checks} / {risk.total_checks}")
summary_cols[3].metric("Unique findings", risk.unique_findings)
st.caption(f"Assessment {result.assessment_id} · {result.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}")
st.caption(risk.methodology)
for limitation in risk.limitations:
    st.caption(limitation)

category_names = list(risk.category_scores)
if category_names:
    st.markdown("**Category risk scores**")
    category_cols = st.columns(len(category_names))
    for column, name in zip(category_cols, category_names):
        column.metric(name.replace("_", " ").title(), f"{risk.category_scores[name]:.1f} / 100",
                      help=f"Weight among evaluated categories: {risk.category_weights[name]:.0%}")
else:
    st.info("No audit category completed checks. The overall risk score is withheld.")

st.subheader("Check results")
status_labels = {
    "passed": "Passed", "finding": "Finding detected", "failed": "Failed to execute",
    "not_applicable": "Not applicable", "not_evaluated": "Not evaluated",
}
check_rows = [{
    "Category": check.category.value.title(), "Check": check.check_id,
    "Status": status_labels[check.status.value], "Findings": len(check.findings),
    "Summary": check.summary,
} for check in result.checks]
st.dataframe(pd.DataFrame(check_rows), width="stretch", hide_index=True)
if result.failed_stages:
    st.warning("Some stages could not run: " + ", ".join(result.failed_stages))

finding_rows = []
for check in result.checks:
    for finding in check.findings:
        finding_rows.append({
            "Severity": finding.severity.value.title(), "Category": finding.category.value.title(),
            "Finding": finding.title, "Evidence": json.dumps(finding.evidence, ensure_ascii=False),
            "Why it matters": finding.explanation, "Source check": finding.source_check,
            "Heuristic": finding.heuristic,
        })
st.subheader("Evidence-backed findings")
if finding_rows:
    st.dataframe(pd.DataFrame(finding_rows), width="stretch", hide_index=True)
else:
    st.info("No findings were produced by completed checks. This is not evidence that the model is safe or production-ready.")

st.subheader("Remediation recommendations")
if result.recommendations:
    for recommendation in result.recommendations:
        with st.container(border=True):
            st.markdown(f"**{recommendation.priority.value.title()} priority · {recommendation.title}**")
            st.write(recommendation.guidance)
            st.caption(f"Why this was generated: {recommendation.rationale}")
            st.caption(f"Finding(s): {', '.join(recommendation.finding_ids)} · Recommendations require validation; no fix is claimed.")
else:
    st.info("Recommendations appear when findings provide evidence for a specific action.")

st.subheader("Consolidated report")
st.caption("Exports contain selected metadata and sanitized findings. Raw dataset rows, prompts and outputs are excluded.")
download_cols = st.columns(2)
download_cols[0].download_button(
    "Download JSON report", json_report_bytes(result),
    file_name=f"trustlens-audit-{result.assessment_id[:8]}.json",
    mime="application/json", width="stretch",
)
download_cols[1].download_button(
    "Download CSV report", csv_report_bytes(result),
    file_name=f"trustlens-audit-{result.assessment_id[:8]}.csv",
    mime="text/csv", width="stretch",
)
st.caption("Human review is required before consequential decisions. No model is automatically changed or deployed.")
