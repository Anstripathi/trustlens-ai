"""Input/output policy enforcement over existing local TrustLens detectors."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from src.security.pii_scanner import PII_CATEGORIES, scan_pii
from src.security.prompt_injection import scan_prompt_injection
from src.security.schemas import Finding, ScanError, Severity, validate_text
from src.security.toxicity import TOXICITY_CATEGORIES, scan_unsafe_output

from .rate_limit import RateLimiter
from .redaction import redact_phrases, redact_sensitive_text
from .schemas import Action, Direction, FirewallPolicy, FirewallResult, PolicyRule, validate_policy

_LEVEL = {Severity.INFORMATIONAL: 0, Severity.LOW: 1, Severity.MEDIUM: 2,
          Severity.HIGH: 3, Severity.CRITICAL: 4}
_ACTION_RANK = {Action.ALLOW: 0, Action.WARN: 1, Action.REDACT: 2,
                Action.APPROVAL_REQUIRED: 3, Action.BLOCK: 4}


def _match_policy_rules(text: str, direction: Direction, rules: tuple[PolicyRule, ...]) -> list[Finding]:
    matches: list[Finding] = []
    for rule in rules:
        if direction not in rule.directions:
            continue
        for phrase in rule.phrases:
            found = phrase in text if rule.case_sensitive else phrase.casefold() in text.casefold()
            if found:
                matches.append(Finding(
                    category="custom_policy", rule_id=rule.rule_id, severity=rule.severity,
                    detector="Configured literal-phrase rule", strength="Configured",
                    explanation=f"Configured policy rule matched ({rule.action.value}).",
                    evidence="Matched policy phrase omitted.", remediation="Apply the configured rule and review the content if needed.",
                    source=direction.value.title(),
                ))
    return matches


def _action_for_finding(finding: Finding, direction: Direction, policy: FirewallPolicy) -> Action:
    if finding.category == "custom_policy":
        rule = next((item for item in policy.custom_rules if item.rule_id == finding.rule_id), None)
        if rule is not None:
            return rule.action
    if finding.category == "prompt_injection":
        rank = _LEVEL[finding.severity]
        return Action.BLOCK if rank >= _LEVEL[policy.block_at] else (
            Action.WARN if rank >= _LEVEL[policy.warn_at] else Action.ALLOW)
    if finding.category in PII_CATEGORIES:
        return policy.input_pii_action if direction == Direction.INPUT else policy.output_pii_action
    rank = _LEVEL[finding.severity]
    if rank >= _LEVEL[policy.block_at]:
        return Action.BLOCK
    if rank >= _LEVEL[policy.warn_at]:
        return Action.WARN
    return Action.ALLOW


def inspect_text(text: str, *, direction: Direction | str, policy: FirewallPolicy | None = None,
                 rate_limiter: RateLimiter | None = None, client_id: str | None = None) -> FirewallResult:
    """Scan once and return an enforcement decision; callers must enforce it before use.

    The returned ``output_text`` is transient payload for the caller and must not be
    serialized or put in event history. A blocked or approval-required result withholds it.
    """
    configured = validate_policy(policy or FirewallPolicy())
    try:
        selected_direction = Direction(direction)
    except (ValueError, TypeError) as exc:
        raise ValueError("Direction must be input or output.") from exc
    now = datetime.now(timezone.utc).isoformat()

    def denied_finding(rule_id: str, category: str, explanation: str) -> Finding:
        return Finding(category=category, rule_id=rule_id, severity=Severity.HIGH,
                       detector="Firewall control", strength="Deterministic", explanation=explanation,
                       evidence="Content omitted.", remediation="Review the policy or request rate and retry when appropriate.",
                       source=selected_direction.value.title())

    if rate_limiter is not None:
        if not client_id:
            raise ValueError("client_id is required when rate limiting is enabled.")
        try:
            permitted = rate_limiter.allow(client_id, limit=configured.max_requests,
                                           window_seconds=configured.window_seconds)
        except Exception:
            permitted = False
        if not permitted:
            return FirewallResult(selected_direction, Action.BLOCK,
                                  (denied_finding("firewall.rate_limited", "operational",
                                                  "The configured request limit was reached or the limiter failed safely."),),
                                  None, configured.policy_id, configured.version, now, rate_limited=True)
    try:
        source_text = validate_text(text)
    except ScanError:
        return FirewallResult(selected_direction, Action.BLOCK,
                              (denied_finding("firewall.invalid_input", "operational",
                                              "Input was invalid or exceeded the local scanner limit."),),
                              None, configured.policy_id, configured.version, now, scan_failed=True)

    try:
        findings = scan_pii(source_text, enabled_categories=configured.pii_categories,
                            custom_labels=configured.sensitive_labels)
        if selected_direction == Direction.INPUT:
            findings.extend(scan_prompt_injection(source_text))
        else:
            findings.extend(scan_unsafe_output(source_text, enabled_categories=configured.unsafe_categories))
        findings.extend(_match_policy_rules(source_text, selected_direction, configured.custom_rules))
    except Exception:
        return FirewallResult(selected_direction, Action.BLOCK,
                              (denied_finding("firewall.scan_failure", "operational",
                                              "A local scanner failed; the firewall blocked the content safely."),),
                              None, configured.policy_id, configured.version, now, scan_failed=True)

    actions = [_action_for_finding(item, selected_direction, configured) for item in findings]
    action = max(actions, key=lambda item: _ACTION_RANK[item], default=Action.ALLOW)
    output: str | None
    if action in (Action.BLOCK, Action.APPROVAL_REQUIRED):
        output = None
    elif action == Action.REDACT:
        output = redact_sensitive_text(source_text)
        for rule in configured.custom_rules:
            if rule.action == Action.REDACT and selected_direction in rule.directions:
                output = redact_phrases(output, rule.phrases, case_sensitive=rule.case_sensitive)
    else:
        output = source_text
    return FirewallResult(selected_direction, action, tuple(findings), output,
                          configured.policy_id, configured.version, now)


@dataclass
class Firewall:
    """Reusable local API wrapper with optional process-local rate limiting."""
    policy: FirewallPolicy = FirewallPolicy()
    rate_limiter: RateLimiter | None = None

    def __post_init__(self) -> None:
        validate_policy(self.policy)

    def inspect(self, text: str, *, direction: Direction | str, client_id: str | None = None) -> FirewallResult:
        return inspect_text(text, direction=direction, policy=self.policy,
                            rate_limiter=self.rate_limiter, client_id=client_id)
