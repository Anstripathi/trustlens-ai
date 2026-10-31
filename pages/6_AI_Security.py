"""Local heuristic security screening for prompts, outputs, and selected data."""
import pandas as pd
import streamlit as st

from src.security.adversarial import CHECK_NAMES, MAX_DATASET_CHARS, run_assessment
from src.security.jailbreak import JAILBREAK_CATEGORIES
from src.security.pii_scanner import PII_CATEGORIES
from src.security.reporting import csv_report_bytes, json_report_bytes
from src.security.schemas import MAX_DATASET_CELLS, MAX_SECURITY_INPUT_CHARS, ScanError, Severity
from src.security.toxicity import TOXICITY_CATEGORIES
from src.theme import record_activity, render_page_header, render_sidebar

st.set_page_config(page_title="AI Security", page_icon=":material/shield_lock:", layout="wide")
render_sidebar()
render_page_header(
    "AI Security & Red Teaming",
    "Run bounded, local heuristic checks over prompts, model outputs, and selected dataset values.",
)
st.info(
    "This page does not call an LLM or execute scanned text. Rules can miss issues and produce false positives; "
    "results are screening signals, not a security verdict. Scanned content stays in this Streamlit session."
)
st.caption("Results are automated and unreviewed. The page does not run model-based moderation or claim target-LLM vulnerability.")

dataset = st.session_state.get("df")
if not isinstance(dataset, pd.DataFrame):
    dataset = None
dataset_options = list(dataset.columns) if dataset is not None else []
assessment = None

with st.form("security_assessment_form"):
    prompt = st.text_area(
        "Prompt or test input",
        max_chars=MAX_SECURITY_INPUT_CHARS,
        height=150,
        help=f"Processed locally. Limit: {MAX_SECURITY_INPUT_CHARS:,} characters.",
        key="security_prompt_input",
    )
    model_output = st.text_area(
        "Optional model output",
        max_chars=MAX_SECURITY_INPUT_CHARS,
        height=150,
        help="Toxicity and PII checks inspect this text when supplied.",
        key="security_model_output_input",
    )
    selected_checks = st.multiselect(
        "Checks to run", CHECK_NAMES, default=list(CHECK_NAMES),
        format_func=lambda name: {
            "prompt_injection": "Prompt injection",
            "jailbreak": "Jailbreak test suite",
            "pii": "PII and sensitive data",
            "toxicity": "Unsafe-output rules",
        }[name],
    )
    with st.expander("Configure test categories and optional dataset scan"):
        jailbreak_categories = st.multiselect(
            "Jailbreak categories", JAILBREAK_CATEGORIES, default=list(JAILBREAK_CATEGORIES),
        )
        pii_categories = st.multiselect(
            "Sensitive-data patterns", PII_CATEGORIES, default=list(PII_CATEGORIES),
        )
        pii_custom_labels_raw = st.text_input(
            "Optional sensitive field labels (comma-separated)",
            placeholder="employee_id, case_number",
            help="Matches simple label=value assignments. Custom regex is not accepted.",
        )
        pii_custom_labels = tuple(label.strip() for label in pii_custom_labels_raw.split(",") if label.strip())
        toxicity_categories = st.multiselect(
            "Unsafe-output categories", TOXICITY_CATEGORIES, default=list(TOXICITY_CATEGORIES),
        )
        scan_dataset = st.checkbox(
            "Also scan selected columns from the currently loaded dataset",
            disabled=dataset is None,
        )
        if dataset is not None:
            dataset_columns = st.multiselect(
                "Dataset columns to scan", dataset_options,
                default=[name for name in dataset_options if dataset[name].dtype == "object"][:5],
            )
        else:
            dataset_columns = []
            st.caption("Load a dataset in Data Explorer to enable this option.")
        dataset_max_cells = st.slider("Maximum dataset cells to examine", 1, MAX_DATASET_CELLS,
                                      min(500, MAX_DATASET_CELLS), 25)
        st.caption(f"Dataset scan has an additional {MAX_DATASET_CHARS:,}-character total budget.")
    submitted = st.form_submit_button("Run local assessment", type="primary")

if submitted:
    st.session_state.pop("security_assessment", None)
    try:
        with st.spinner("Running selected local checks..."):
            assessment = run_assessment(
                prompt=prompt,
                model_output=model_output,
                selected_checks=selected_checks,
                jailbreak_categories=jailbreak_categories,
                pii_categories=pii_categories,
                pii_custom_labels=pii_custom_labels,
                toxicity_categories=toxicity_categories,
                dataset=dataset if scan_dataset and "pii" in selected_checks else None,
                dataset_columns=dataset_columns if scan_dataset and "pii" in selected_checks else (),
                dataset_max_cells=dataset_max_cells,
            )
        st.session_state["security_assessment"] = assessment
        risk_label = assessment["risk"].get("risk_level", "Not assessed")
        record_activity("Security assessment run", f"{risk_label} · {len(assessment['findings'])} finding(s)")
    except ScanError as exc:
        st.error(str(exc))
    except Exception:  # noqa: BLE001 - do not echo scanner or input details
        st.error("The assessment could not be completed. Check the selected inputs and try again.")

assessment = assessment or st.session_state.get("security_assessment")
if assessment is None:
    st.subheader("Assessment results")
    st.info("Select the checks you need, add a prompt or output, and run an assessment to see findings here.")
    st.stop()

st.divider()
st.subheader("Assessment summary")
summary_cols = st.columns(3)
summary_cols[0].metric("Overall assessment", assessment["risk"]["risk_level"])
summary_cols[1].metric("Risk score", "n/a" if assessment["risk"]["score"] is None else f"{assessment['risk']['score']:.1f} / 100")
summary_cols[2].metric("Findings", len(assessment["findings"]))
st.caption(
    f"Assessment {assessment['assessment_id']} · {assessment['created_at']} · "
    f"status: {assessment['status']}"
)
st.caption(assessment["risk"]["methodology"])
st.warning(assessment["risk"].get("limitations", "A low score does not establish that content is secure."))

check_rows = [{"Check": name.replace("_", " ").title(), **result}
              for name, result in assessment["checks"].items()]
st.markdown("**Check status**")
st.dataframe(pd.DataFrame(check_rows).drop(columns=["case_results"], errors="ignore"),
             use_container_width=True, hide_index=True)

for check_name, check in assessment["checks"].items():
    cases = check.get("case_results", [])
    if cases:
        with st.expander(f"Jailbreak test cases ({len(cases)})"):
            st.caption("These safe, built-in prompts exercise local pattern rules. No target LLM is called.")
            st.dataframe(pd.DataFrame(cases), use_container_width=True, hide_index=True)

st.markdown("**Findings by severity**")
severity_cols = st.columns(len(Severity))
severity_counts = assessment["risk"].get("severity_counts", {})
for col, severity in zip(severity_cols, Severity):
    col.metric(severity.value, severity_counts.get(severity.value, 0), border=True)

if assessment["findings"]:
    findings_frame = pd.DataFrame([finding.to_dict() for finding in assessment["findings"]])
    st.dataframe(
        findings_frame[["source", "category", "severity", "rule_id", "detector", "strength", "human_review_status", "evidence"]],
        use_container_width=True,
        hide_index=True,
    )
    for severity in Severity:
        matches = [finding for finding in assessment["findings"] if finding.severity == severity]
        if not matches:
            continue
        with st.expander(f"{severity.value} findings ({len(matches)})"):
            for finding in matches:
                st.markdown(f"**{finding.category.replace('_', ' ').title()} · {finding.source}**")
                st.caption(f"{finding.rule_id} · {finding.detector} · strength: {finding.strength}")
                st.write(finding.explanation)
                st.caption(f"Evidence: {finding.evidence}")
                st.write(f"Recommended action: {finding.remediation}")
else:
    st.success("No findings were produced by the completed checks.")
    st.caption("This is not proof that the scanned content is safe; review which checks ran and their limits.")

st.subheader("Export assessment")
json_data = json_report_bytes(assessment)
csv_data = csv_report_bytes(assessment)
export_cols = st.columns(2)
export_cols[0].download_button("Download redacted JSON", json_data,
                               file_name=f"trustlens-security-{assessment['assessment_id'][:8]}.json",
                               mime="application/json", use_container_width=True)
export_cols[1].download_button("Download redacted CSV", csv_data,
                               file_name=f"trustlens-security-{assessment['assessment_id'][:8]}.csv",
                               mime="text/csv", use_container_width=True)
st.caption("Exports contain assessment metadata and masked findings only; original prompts, outputs, and dataset values are excluded.")
