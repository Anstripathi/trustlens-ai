"""Shared visual language and page chrome for TrustLens AI."""
from __future__ import annotations

from datetime import datetime

import streamlit as st


BRAND = "TrustLens AI"
PAGES = (
    ("Data Explorer", "pages/1_Data_Explorer.py", "table_chart"),
    ("Model Lab", "pages/2_Model_Lab.py", "model_training"),
    ("Explainability", "pages/4_Explainability.py", "query_stats"),
    ("Reliability Monitor", "pages/5_Reliability_Monitor.py", "verified_user"),
    ("AI Security", "pages/6_AI_Security.py", "shield_lock"),
    ("Model Auditor", "pages/7_Model_Auditor.py", "policy"),
    ("AI Firewall", "pages/8_AI_Firewall.py", "shield_lock"),
    ("Model Monitoring", "pages/9_Model_Monitoring.py", "monitoring"),
    ("Advanced Model Intelligence", "pages/10_Advanced_Model_Intelligence.py", "insights"),
    ("Governance Center", "pages/11_Governance_Center.py", "account_tree"),
    ("LLM and RAG Evaluation", "pages/12_LLM_RAG_Evaluation.py", "psychology"),
    ("Governance Control Center", "pages/13_Governance_Control_Center.py", "rule"),
)


def render_sidebar() -> None:
    """Add consistent brand context above Streamlit's native page navigation."""
    with st.sidebar:
        st.markdown(f"### {BRAND}")
        st.caption("MODEL INTELLIGENCE · GOVERNANCE")
        st.divider()
        st.caption("Analysis runs in your session. Uploaded data is not saved by the app.")


def render_page_header(title: str, description: str, *, eyebrow: str = "TRUSTLENS AI") -> None:
    """Render a consistent, quiet page heading across application modules."""
    st.caption(eyebrow)
    st.title(title)
    st.caption(description)


def record_activity(action: str, detail: str) -> None:
    """Keep a small, session-only record of actions the app actually performed."""
    events = list(st.session_state.get("activity", []))
    events.insert(0, {"action": action, "detail": detail, "at": datetime.now().astimezone()})
    st.session_state["activity"] = events[:8]
