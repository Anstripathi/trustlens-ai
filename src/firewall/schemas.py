"""Validated, JSON-safe policy and result types for the local firewall."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re

from src.security.pii_scanner import PII_CATEGORIES
from src.security.schemas import Finding, Severity
from src.security.toxicity import TOXICITY_CATEGORIES


class Direction(str, Enum):
    INPUT = "input"
    OUTPUT = "output"


class Action(str, Enum):
    ALLOW = "allow"
    WARN = "warn"
    BLOCK = "block"
    REDACT = "redact"
    APPROVAL_REQUIRED = "approval_required"


@dataclass(frozen=True)
class PolicyRule:
    """Match literal phrases only; policy text is never compiled or executed."""

    rule_id: str
    phrases: tuple[str, ...]
    action: Action
    directions: tuple[Direction, ...] = (Direction.INPUT, Direction.OUTPUT)
    severity: Severity = Severity.HIGH
    case_sensitive: bool = False


@dataclass(frozen=True)
class FirewallPolicy:
    policy_id: str = "trustlens-default"
    version: str = "1"
    block_at: Severity = Severity.HIGH
    warn_at: Severity = Severity.MEDIUM
    input_pii_action: Action = Action.WARN
    output_pii_action: Action = Action.REDACT
    max_requests: int = 60
    window_seconds: int = 60
    pii_categories: tuple[str, ...] | None = None
    unsafe_categories: tuple[str, ...] | None = None
    sensitive_labels: tuple[str, ...] = ()
    custom_rules: tuple[PolicyRule, ...] = ()


@dataclass(frozen=True)
class FirewallResult:
    direction: Direction
    action: Action
    findings: tuple[Finding, ...]
    output_text: str | None
    policy_id: str
    policy_version: str
    checked_at: str
    rate_limited: bool = False
    scan_failed: bool = False

    @property
    def requires_approval(self) -> bool:
        return self.action == Action.APPROVAL_REQUIRED


def validate_policy(policy: FirewallPolicy) -> FirewallPolicy:
    """Reject malformed or unbounded policy data before scanning any content."""
    if not isinstance(policy, FirewallPolicy):
        raise ValueError("A FirewallPolicy instance is required.")
    if not isinstance(policy.policy_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", policy.policy_id):
        raise ValueError("Policy ID must be 1–80 letters, digits, underscores, or hyphens.")
    if not isinstance(policy.version, str) or not 1 <= len(policy.version) <= 40:
        raise ValueError("Policy version must be 1–40 characters.")
    if not isinstance(policy.block_at, Severity) or not isinstance(policy.warn_at, Severity):
        raise ValueError("Policy thresholds must use a supported severity.")
    levels = {Severity.INFORMATIONAL: 0, Severity.LOW: 1, Severity.MEDIUM: 2,
              Severity.HIGH: 3, Severity.CRITICAL: 4}
    if levels[policy.warn_at] > levels[policy.block_at]:
        raise ValueError("Warning threshold cannot be more restrictive than the block threshold.")
    if not isinstance(policy.input_pii_action, Action) or not isinstance(policy.output_pii_action, Action):
        raise ValueError("PII actions must use a supported firewall action.")
    if not 1 <= policy.max_requests <= 10_000 or not 1 <= policy.window_seconds <= 86_400:
        raise ValueError("Rate limits must allow 1–10,000 requests per 1–86,400 seconds.")
    if policy.pii_categories is not None and set(policy.pii_categories) - set(PII_CATEGORIES):
        raise ValueError("Policy contains an unsupported sensitive-data category.")
    if policy.unsafe_categories is not None and set(policy.unsafe_categories) - set(TOXICITY_CATEGORIES):
        raise ValueError("Policy contains an unsupported unsafe-output category.")
    if len(policy.custom_rules) > 50:
        raise ValueError("A policy may contain at most 50 custom rules.")
    ids: set[str] = set()
    for rule in policy.custom_rules:
        if not isinstance(rule, PolicyRule):
            raise ValueError("Custom rules must use the PolicyRule type.")
        if not rule.rule_id or len(rule.rule_id) > 80 or rule.rule_id in ids:
            raise ValueError("Custom rule IDs must be unique and 1–80 characters.")
        ids.add(rule.rule_id)
        if not isinstance(rule.action, Action) or not isinstance(rule.severity, Severity):
            raise ValueError("Custom rule action and severity are invalid.")
        if not isinstance(rule.case_sensitive, bool):
            raise ValueError("Custom rule case_sensitive must be boolean.")
        if not rule.directions or set(rule.directions) - set(Direction):
            raise ValueError("Custom rules need one or more supported directions.")
        if not 1 <= len(rule.phrases) <= 20:
            raise ValueError("Each custom rule needs 1–20 literal phrases.")
        if any(not isinstance(phrase, str) or not phrase.strip() or len(phrase) > 300 for phrase in rule.phrases):
            raise ValueError("Custom phrases must be non-empty strings of at most 300 characters.")
    if len(policy.sensitive_labels) > 6 or any(
        not isinstance(label, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,31}", label)
        for label in policy.sensitive_labels
    ):
        raise ValueError("At most six text labels can be configured for sensitive data.")
    return policy
