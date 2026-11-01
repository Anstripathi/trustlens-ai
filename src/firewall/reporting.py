"""Exports for firewall policy and event metadata (never scanned payloads)."""
from __future__ import annotations

import csv
import io
import json

from .schemas import FirewallPolicy


def _safe(value) -> str:
    text = "" if value is None else str(value)
    return "'" + text if text.lstrip().startswith(("=", "+", "-", "@", "\t", "\r")) else text


def build_report(*, policy: FirewallPolicy, events: list[dict], approvals: list[dict]) -> dict:
    return {
        "report_version": 1,
        "policy": {
            "policy_id": policy.policy_id, "version": policy.version,
            "block_at": policy.block_at.value, "warn_at": policy.warn_at.value,
            "input_pii_action": policy.input_pii_action.value,
            "output_pii_action": policy.output_pii_action.value,
            "max_requests": policy.max_requests, "window_seconds": policy.window_seconds,
            "custom_rule_ids": [rule.rule_id for rule in policy.custom_rules],
        },
        "events": [{key: event.get(key) for key in (
            "event_id", "created_at", "direction", "action", "policy_id", "policy_version",
            "finding_count", "categories", "rule_ids", "rate_limited", "scan_failed")}
            for event in events],
        "approvals": [{key: item.get(key) for key in (
            "approval_id", "requested_at", "direction", "policy_id", "policy_version",
            "rule_ids", "status", "decided_at", "reviewer")}
            for item in approvals],
        "limitations": [
            "Deterministic local rules are screening controls, not guaranteed protection.",
            "Events contain metadata only; scanned content is intentionally excluded.",
            "Approval records are session-local and do not authenticate reviewer identity.",
        ],
    }


def json_report_bytes(*, policy: FirewallPolicy, events: list[dict], approvals: list[dict]) -> bytes:
    return json.dumps(build_report(policy=policy, events=events, approvals=approvals),
                      indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8")


def csv_report_bytes(*, policy: FirewallPolicy, events: list[dict], approvals: list[dict]) -> bytes:
    report = build_report(policy=policy, events=events, approvals=approvals)
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(["record_type", "record_id", "created_at", "direction", "action_or_status",
                     "policy_id", "policy_version", "finding_count", "categories", "rule_ids", "reviewer"])
    for event in report["events"]:
        writer.writerow([_safe("event"), *[_safe(event.get(k)) for k in (
            "event_id", "created_at", "direction", "action", "policy_id", "policy_version",
            "finding_count", "categories", "rule_ids")], ""])
    for item in report["approvals"]:
        writer.writerow([_safe("approval"), *[_safe(item.get(k)) for k in (
            "approval_id", "requested_at", "direction", "status", "policy_id", "policy_version")],
            "", "", _safe(item.get("rule_ids")), _safe(item.get("reviewer"))])
    return output.getvalue().encode("utf-8")
