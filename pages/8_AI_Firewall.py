"""Local AI Firewall test console and session-scoped policy management."""
from __future__ import annotations

from dataclasses import replace
import json
import uuid

import streamlit as st

from src.firewall.approvals import ApprovalQueue
from src.firewall.engine import Firewall
from src.firewall.rate_limit import RateLimiter
from src.firewall.reporting import csv_report_bytes, json_report_bytes
from src.firewall.schemas import Action, Direction, FirewallPolicy, PolicyRule
from src.security.schemas import Severity
from src.theme import render_page_header, render_sidebar

st.set_page_config(page_title="AI Firewall", page_icon=":material/shield_lock:", layout="wide")
render_sidebar()
render_page_header("AI Firewall & Guardrails", "Apply local input and output policies before content reaches an AI system or user.")
st.info("Local heuristic checks only. This page never calls an AI model or transmits content. Integrations must enforce each returned decision; scanners can miss issues and produce false positives.")

if "firewall_policy" not in st.session_state:
    st.session_state.firewall_policy = FirewallPolicy()
if "firewall_policy_history" not in st.session_state:
    st.session_state.firewall_policy_history = [{"version": st.session_state.firewall_policy.version, "reason": "Initial default policy"}]
if "firewall_events" not in st.session_state:
    st.session_state.firewall_events = []
if "firewall_approval_queue" not in st.session_state:
    st.session_state.firewall_approval_queue = ApprovalQueue()
if "firewall_limiter" not in st.session_state:
    st.session_state.firewall_limiter = RateLimiter()
if "firewall_client_id" not in st.session_state:
    st.session_state.firewall_client_id = str(uuid.uuid4())

policy: FirewallPolicy = st.session_state.firewall_policy
severity_order = [Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL]
with st.expander("Policy controls", expanded=True):
    with st.form("firewall_policy_form"):
        block_at = st.selectbox("Block content at or above severity", severity_order,
                                index=severity_order.index(policy.block_at) if policy.block_at in severity_order else 1,
                                format_func=lambda item: item.value)
        warn_at = st.selectbox("Warn content at or above severity", [Severity.LOW, Severity.MEDIUM],
                               index=0 if policy.warn_at == Severity.LOW else 1,
                               format_func=lambda item: item.value)
        pii_input_action = st.selectbox("Input PII action", [Action.WARN, Action.BLOCK, Action.REDACT],
                                        index=[Action.WARN, Action.BLOCK, Action.REDACT].index(policy.input_pii_action),
                                        format_func=lambda item: item.value.replace("_", " ").title())
        pii_output_action = st.selectbox("Output PII action", [Action.REDACT, Action.BLOCK, Action.WARN],
                                         index=[Action.REDACT, Action.BLOCK, Action.WARN].index(policy.output_pii_action),
                                         format_func=lambda item: item.value.replace("_", " ").title())
        max_requests = st.number_input("Requests per session window", min_value=1, max_value=10_000, value=policy.max_requests, step=1)
        window_seconds = st.number_input("Rate-limit window (seconds)", min_value=1, max_value=86_400, value=policy.window_seconds, step=1)
        save_policy = st.form_submit_button("Save policy version", type="primary")
    if save_policy:
        version = str(int(policy.version) + 1) if policy.version.isdigit() else "2"
        st.session_state.firewall_policy = replace(policy, version=version, block_at=block_at, warn_at=warn_at,
                                                   input_pii_action=pii_input_action, output_pii_action=pii_output_action,
                                                   max_requests=int(max_requests), window_seconds=int(window_seconds))
        st.session_state.firewall_policy_history.append({"version": version, "reason": "Policy controls updated"})
        st.rerun()

policy = st.session_state.firewall_policy
st.caption(f"Active policy: `{policy.policy_id}` · version `{policy.version}` · rate limit {policy.max_requests}/{policy.window_seconds}s")
with st.expander("Policy version history"):
    st.dataframe(st.session_state.firewall_policy_history, width="stretch", hide_index=True)

with st.expander("Custom literal rules"):
    st.caption("Use literal phrases only. Custom policies do not accept or execute regular expressions.")
    with st.form("firewall_custom_rule_form", clear_on_submit=True):
        phrase = st.text_input("Phrase to match", max_chars=300)
        rule_direction = st.selectbox("Applies to", ["input", "output", "both"])
        rule_action = st.selectbox("Action", [Action.WARN, Action.BLOCK, Action.REDACT, Action.APPROVAL_REQUIRED],
                                   format_func=lambda item: item.value.replace("_", " ").title())
        rule_severity = st.selectbox("Finding severity", [Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL],
                                     index=2, format_func=lambda item: item.value)
        add_rule = st.form_submit_button("Add policy rule")
    if add_rule:
        if not phrase.strip():
            st.error("Enter a literal phrase.")
        elif len(policy.custom_rules) >= 50:
            st.error("A policy can contain at most 50 rules.")
        else:
            directions = (Direction.INPUT, Direction.OUTPUT) if rule_direction == "both" else (Direction(rule_direction),)
            rule = PolicyRule(f"custom-{uuid.uuid4().hex[:10]}", (phrase.strip(),), rule_action, directions, rule_severity)
            version = str(int(policy.version) + 1) if policy.version.isdigit() else "2"
            st.session_state.firewall_policy = replace(policy, version=version, custom_rules=(*policy.custom_rules, rule))
            st.session_state.firewall_policy_history.append({"version": version, "reason": f"Added {rule.rule_id}"})
            st.rerun()
    if policy.custom_rules:
        st.dataframe([{"Rule": rule.rule_id, "Directions": ", ".join(d.value for d in rule.directions),
                       "Action": rule.action.value, "Severity": rule.severity.value, "Phrase count": len(rule.phrases)}
                      for rule in policy.custom_rules], width="stretch", hide_index=True)
        remove_rule = st.selectbox("Remove rule", ["Select a rule", *[rule.rule_id for rule in policy.custom_rules]])
        if remove_rule != "Select a rule" and st.button("Remove selected rule"):
            version = str(int(policy.version) + 1) if policy.version.isdigit() else "2"
            st.session_state.firewall_policy = replace(policy, version=version,
                                                       custom_rules=tuple(rule for rule in policy.custom_rules if rule.rule_id != remove_rule))
            st.session_state.firewall_policy_history.append({"version": version, "reason": f"Removed {remove_rule}"})
            st.rerun()
    else:
        st.caption("No custom phrase rules have been configured.")

st.subheader("Firewall test console")
with st.form("firewall_console_form"):
    direction = st.radio("Inspect content as", [Direction.INPUT, Direction.OUTPUT], horizontal=True,
                         format_func=lambda item: item.value.title())
    content = st.text_area("Content to inspect", max_chars=20_000, height=180,
                           help="Scanned content is held in this session; it is omitted from events and reports.")
    submitted = st.form_submit_button("Inspect and enforce policy", type="primary")

if submitted:
    result = Firewall(policy, st.session_state.firewall_limiter).inspect(
        content, direction=direction, client_id=st.session_state.firewall_client_id)
    event = {"event_id": str(uuid.uuid4()), "created_at": result.checked_at,
             "direction": result.direction.value, "action": result.action.value,
             "policy_id": result.policy_id, "policy_version": result.policy_version,
             "finding_count": len(result.findings),
             "categories": sorted({item.category for item in result.findings}),
             "rule_ids": list(dict.fromkeys(item.rule_id for item in result.findings)),
             "rate_limited": result.rate_limited, "scan_failed": result.scan_failed}
    st.session_state.firewall_events.insert(0, event)
    st.session_state.firewall_events = st.session_state.firewall_events[:100]
    if result.requires_approval:
        st.session_state.firewall_approval_queue.submit(result)
    st.session_state.firewall_latest_result_summary = {
        "action": result.action.value, "direction": result.direction.value,
        "finding_count": len(result.findings),
        "findings": [{"category": f.category, "rule_id": f.rule_id, "severity": f.severity.value,
                      "explanation": f.explanation, "evidence": f.evidence, "remediation": f.remediation}
                     for f in result.findings],
        "redacted_preview": result.output_text if result.action == Action.REDACT else None,
    }

summary = st.session_state.get("firewall_latest_result_summary")
if summary:
    st.divider()
    st.subheader("Latest enforcement decision")
    col1, col2 = st.columns(2)
    col1.metric("Decision", summary["action"].replace("_", " ").title())
    col2.metric("Findings", summary["finding_count"])
    if summary["action"] == Action.BLOCK.value:
        st.error("Blocked. Do not forward this content to the protected system or user.")
    elif summary["action"] == Action.APPROVAL_REQUIRED.value:
        st.warning("Held for human approval. The firewall did not return the scanned payload.")
    elif summary["action"] == Action.REDACT.value:
        st.warning("Sensitive content was redacted before release.")
        st.code(summary.get("redacted_preview") or "")
    elif summary["action"] == Action.WARN.value:
        st.warning("Review the findings before forwarding this content.")
    else:
        st.success("Allowed by the active policy. This is not proof the content is safe.")
    for finding in summary["findings"]:
        with st.expander(f"{finding['severity']} · {finding['category']} · {finding['rule_id']}"):
            st.write(finding["explanation"])
            st.caption(finding["evidence"])
            st.write(finding["remediation"])

st.subheader("Human approval queue")
queue = st.session_state.firewall_approval_queue
records = queue.list()
pending = [record for record in records if record.status == "pending"]
if pending:
    reviewer = st.text_input("Reviewer label", value="Session reviewer", max_chars=64, key="firewall_reviewer")
    for record in pending:
        st.warning(f"Pending {record.direction} action · policy {record.policy_id} v{record.policy_version} · rules: {', '.join(record.rule_ids)}")
        approve_col, reject_col = st.columns(2)
        if approve_col.button("Approve this action", key=f"approve-{record.approval_id}"):
            try:
                queue.decide(record.approval_id, approved=True, reviewer=reviewer)
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
        if reject_col.button("Reject this action", key=f"reject-{record.approval_id}"):
            try:
                queue.decide(record.approval_id, approved=False, reviewer=reviewer)
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
    st.caption("Approval records are session-local; reviewer labels are not authentication.")
elif records:
    st.caption("No pending decisions.")
else:
    st.caption("Custom rules can hold matching content for a human decision.")

st.subheader("Session security events")
events = st.session_state.firewall_events
if events:
    st.dataframe([{"Time": e["created_at"], "Direction": e["direction"], "Decision": e["action"],
                   "Findings": e["finding_count"], "Categories": ", ".join(e["categories"]),
                   "Policy": f"{e['policy_id']} v{e['policy_version']}", "Rate limited": e["rate_limited"]}
                  for e in events], width="stretch", hide_index=True)
else:
    st.caption("No firewall inspections have been recorded in this session.")

st.subheader("Policy and event report")
approvals = [record.to_dict() for record in records]
json_data = json_report_bytes(policy=policy, events=events, approvals=approvals)
csv_data = csv_report_bytes(policy=policy, events=events, approvals=approvals)
policy_json = json.dumps({"policy_id": policy.policy_id, "version": policy.version,
                          "block_at": policy.block_at.value, "warn_at": policy.warn_at.value,
                          "input_pii_action": policy.input_pii_action.value,
                          "output_pii_action": policy.output_pii_action.value,
                          "max_requests": policy.max_requests, "window_seconds": policy.window_seconds,
                          "custom_rules": [{"rule_id": r.rule_id, "phrases": list(r.phrases),
                                            "action": r.action.value, "directions": [d.value for d in r.directions],
                                            "severity": r.severity.value, "case_sensitive": r.case_sensitive}
                                           for r in policy.custom_rules]},
                         ensure_ascii=False, indent=2).encode("utf-8")
cols = st.columns(3)
cols[0].download_button("Download event report (JSON)", json_data, "trustlens-firewall-report.json", "application/json", width="stretch")
cols[1].download_button("Download event report (CSV)", csv_data, "trustlens-firewall-report.csv", "text/csv", width="stretch")
cols[2].download_button("Download current policy (JSON)", policy_json, "trustlens-firewall-policy.json", "application/json", width="stretch")
st.caption("Event reports contain metadata only. Policy JSON contains literal rules you configured.")
