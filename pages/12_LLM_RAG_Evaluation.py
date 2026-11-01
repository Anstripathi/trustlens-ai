"""Offline evaluation for user-provided LLM and RAG traces."""
import streamlit as st

from src.data_validation import ValidationError, load_csv_bytes
from src.llm_eval import evaluate_traces
from src.theme import render_page_header, render_sidebar

st.set_page_config(page_title="LLM and RAG Evaluation", page_icon=":material/psychology:", layout="wide")
render_sidebar()
render_page_header("LLM and RAG Evaluation", "Assess supplied traces with transparent local checks; no model endpoint is called.")
st.caption("Upload a CSV with prompt, answer and context columns. Optional latency_ms is summarized only when supplied.")
upload = st.file_uploader("Evaluation traces", type=["csv"], max_upload_size=25, key="llm_eval_upload")
if upload is None:
    st.info("No trace file uploaded. Supply evaluation data to calculate actual results.")
    st.stop()
try:
    frame = load_csv_bytes(upload.getvalue(), upload.name).clean
    report = evaluate_traces(frame)
except (ValidationError, ValueError) as exc:
    st.error(str(exc))
    st.stop()

st.subheader("Evaluation summary")
cols = st.columns(4)
cols[0].metric("Rows evaluated", report["rows_evaluated"])
cols[1].metric("Lexical grounding overlap", f"{report['mean_lexical_grounding_overlap']:.1%}")
cols[2].metric("Citation markers present", f"{report['citation_presence_rate']:.1%}")
cols[3].metric("Injection patterns flagged", report["prompt_injection_flag_count"])
cols2 = st.columns(3)
cols2[0].metric("PII patterns in answers", report["output_pii_finding_count"])
cols2[1].metric("Repeated prompt answer variance", report["repeated_prompt_answer_variance_count"])
cols2[2].metric("Mean supplied latency", "Not supplied" if report["mean_latency_ms"] is None else f"{report['mean_latency_ms']:.1f} ms")
for limitation in report["limitations"]:
    st.caption(f"Limitation: {limitation}")
st.success("Evaluation completed in memory. Prompt, answer and context text are omitted from the results and are not persisted by this page.")
