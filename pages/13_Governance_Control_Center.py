"""Versioned local governance policies and audit log review."""
import json

import streamlit as st

from src import config
from src.governance_records import AuditTrail, PolicyRegistry, PolicySnapshot
from src.theme import render_page_header, render_sidebar

st.set_page_config(page_title="Governance Control Center", page_icon=":material/rule:", layout="wide")
render_sidebar()
render_page_header("Governance Control Center", "Manage versioned policy snapshots and review the local governance event trail.")
registry = PolicyRegistry(config.ARTIFACTS_DIR / "governance")
trail = AuditTrail(config.ARTIFACTS_DIR / "governance")

st.subheader("Publish a policy version")
with st.form("policy_form"):
    policy_name = st.text_input("Policy name", value="Default local review policy", max_chars=80)
    risk_threshold = st.slider("Maximum risk score", 0, 100, 35)
    require_review = st.checkbox("Require human review for model use", value=True)
    actor = st.text_input("Actor label", max_chars=100)
    claimed_role = st.selectbox("Claimed role", ["admin", "analyst", "reviewer", "viewer"])
    publish = st.form_submit_button("Publish next version", type="primary")
if publish:
    try:
        if claimed_role != "admin" or not actor.strip():
            raise ValueError("Publishing a policy requires a non-empty actor label with the claimed admin role.")
        existing = [p for p in registry.list() if p.get("name") == policy_name]
        version = max((int(p.get("version", 0)) for p in existing), default=0) + 1
        policy = PolicySnapshot(policy_name, version, float(risk_threshold), require_review)
        published = registry.publish(policy)
        trail.append(actor=actor, role=claimed_role, action="publish_policy",
                     object_id=f"{policy_name}:v{version}", policy_digest=policy.digest, outcome="published")
        st.success(f"Published policy version {version} · SHA-256 {policy.digest[:16]}…")
    except (ValueError, OSError) as exc:
        st.error(str(exc))

st.subheader("Versioned policies")
policies = registry.list()
if policies:
    st.dataframe([{k: v for k, v in p.items() if k not in {"allowed_actions"}} for p in policies],
                 use_container_width=True, hide_index=True)
else:
    st.info("No policy versions have been published in this workspace.")

st.subheader("Governance audit trail")
try:
    events = trail.events()
    if events:
        st.dataframe([{k: v for k, v in e.items() if k != "previous_hash"} for e in events],
                     use_container_width=True, hide_index=True)
        st.download_button("Download audit trail JSON", json.dumps(events, indent=2),
                           file_name="trustlens-governance-audit.json", mime="application/json")
        st.caption("Hash chain verification passed for the records currently readable from local storage.")
    else:
        st.info("No policy or governance events have been recorded yet.")
except (ValueError, OSError) as exc:
    st.error(f"Audit trail verification failed: {exc}")

st.warning("Role names are user-provided claims; this Streamlit app does not authenticate identities. The hash chain can reveal accidental edits but cannot prevent a local administrator or storage host from replacing the whole log. This is not a certified or tamper-proof audit system.")
