"""Bounded audit orchestration with explicit human approval records."""
import pandas as pd
import streamlit as st

from src.auditor import AuditRequest
from src.artifact_manager import dataset_fingerprint
from src.governance import GovernancePolicy, GovernanceQueue, policy_gate, run_governance
from src.theme import record_activity, render_page_header, render_sidebar

st.set_page_config(page_title="Governance Center", page_icon=":material/account_tree:", layout="wide")
render_sidebar()
render_page_header("Governance Center", "Coordinate bounded local assessments and record human review decisions.")

df = st.session_state.get("df")
if df is None:
    st.info("Load a dataset before running a governance assessment.")
    st.stop()
fp = st.session_state.get("df_fp") or dataset_fingerprint(df)
model = st.session_state.get("result")
current_model = model if model is not None and st.session_state.get("model_data_fp") == fp else None
queue = st.session_state.setdefault("governance_queue", GovernanceQueue())

st.subheader("Assessment policy")
threshold = st.slider("Maximum risk score before mandatory review", 0, 100, 35)
policy = GovernancePolicy(max_risk_score=float(threshold))
st.caption("Thresholds route a run for human review. They do not certify a model or authorize deployment.")
if st.button("Run bounded governance assessment", type="primary"):
    request = AuditRequest(dataset=df, dataset_name=st.session_state.get("dataset_name"),
                           dataset_fingerprint=fp, model=current_model,
                           model_fingerprint=fp if current_model is not None else None,
                           run_security=False)
    with st.spinner("Running the fixed local assessment stages once..."):
        st.session_state["governance_run"] = run_governance(request)
    record_activity("Governance assessment completed", "Fixed local audit stages")

run = st.session_state.get("governance_run")
if run:
    audit = run["audit"]
    gate = policy_gate(audit, policy)
    st.subheader("Assessment result")
    c1, c2, c3 = st.columns(3)
    c1.metric("Routing", gate["decision"].replace("_", " ").title())
    c2.metric("Risk score", "Withheld" if audit.risk.score is None else f"{audit.risk.score:.1f} / 100")
    c3.metric("Assessment stages", f"{len(audit.checks)} checks")
    for reason in gate["reasons"]:
        st.warning(reason)
    st.caption(f"Run {run['run_id']} · {run['orchestration']}")
    checks = [{"Category": c.category.value, "Check": c.check_id, "Status": c.status.value,
               "Findings": len(c.findings), "Summary": c.summary} for c in audit.checks]
    st.dataframe(pd.DataFrame(checks), use_container_width=True, hide_index=True)
    if gate["decision"] == "eligible_for_human_review":
        action = st.selectbox("Request human decision", ["review_model", "approve_for_local_use", "reject_model"],
                              format_func=lambda x: x.replace("_", " ").title())
        if st.button("Create review request"):
            queue.request(run["run_id"], action)
            st.success("Review request recorded in this session.")

st.subheader("Human review queue")
if not queue.records:
    st.info("No review decisions have been requested in this session.")
else:
    for item in queue.records:
        with st.container(border=True):
            st.markdown(f"**{item.action.replace('_', ' ').title()}** · {item.status.title()}")
            st.caption(f"Decision {item.decision_id} · run {item.run_id}")
            if item.status == "pending":
                with st.form(f"decision_{item.decision_id}"):
                    reviewer = st.text_input("Reviewer name", max_chars=100, key=f"reviewer_{item.decision_id}")
                    role = st.selectbox("Claimed role", ["reviewer", "admin", "viewer"], key=f"role_{item.decision_id}")
                    outcome = st.radio("Decision", ["approved", "rejected"], horizontal=True,
                                       key=f"outcome_{item.decision_id}")
                    submitted = st.form_submit_button("Record decision")
                if submitted:
                    try:
                        queue.decide(item.decision_id, reviewer, role, outcome)
                        st.rerun()
                    except ValueError as exc:
                        st.error(str(exc))
            elif item.reviewer:
                st.caption(f"Recorded by {item.reviewer} · {item.decided_at}")

st.warning("Reviewer roles and history are session-local claims; this deployment has no authentication or tamper-proof audit store. No model is modified or deployed by the orchestrator.")
