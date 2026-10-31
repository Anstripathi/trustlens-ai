"""Shared, privacy-conscious structures for local security checks."""
from __future__ import annotations

import re
import ipaddress
from dataclasses import asdict, dataclass
from enum import Enum

MAX_SECURITY_INPUT_CHARS = 20_000
MAX_EVIDENCE_CHARS = 180
MAX_DATASET_CELLS = 2_000


class Severity(str, Enum):
    INFORMATIONAL = "Informational"
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"


class ScanError(ValueError):
    """Safe, actionable input or configuration problem."""


@dataclass(frozen=True)
class Finding:
    category: str
    rule_id: str
    severity: Severity
    detector: str
    strength: str
    explanation: str
    evidence: str
    remediation: str
    source: str = "Input"
    human_review_status: str = "Not reviewed"

    def to_dict(self) -> dict:
        value = asdict(self)
        value["severity"] = self.severity.value
        return value


def validate_text(text: str, *, max_chars: int = MAX_SECURITY_INPUT_CHARS) -> str:
    if not isinstance(text, str):
        raise ScanError("Security checks require text input.")
    if max_chars < 1 or max_chars > MAX_SECURITY_INPUT_CHARS:
        raise ScanError(f"Input limit must be between 1 and {MAX_SECURITY_INPUT_CHARS} characters.")
    if len(text) > max_chars:
        raise ScanError(f"Text exceeds the {max_chars:,}-character security scan limit.")
    return text


_REDACTION_PATTERNS = (
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]{0,6000}(?:-----END [A-Z ]*PRIVATE KEY-----)?", re.I), "[REDACTED PRIVATE KEY]"),
    (re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I), "[REDACTED EMAIL]"),
    (re.compile(r"\b(?:password|passwd|pwd|secret|api[_ -]?key|access[_ -]?token|auth[_ -]?token|authorization)\s*(?:[:=]|\bis\b)\s*[^\s,;]+", re.I), "[REDACTED CREDENTIAL]"),
    (re.compile(r"\b(?:sk_live_|sk_test_|gh[pousr]_|xox[baprs]-)[A-Za-z0-9_-]{8,}\b", re.I), "[REDACTED TOKEN]"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "[REDACTED TOKEN]"),
    (re.compile(r"\b(?=[A-Za-z0-9_./+=:-]{24,}\b)(?=[A-Za-z0-9_./+=:-]*[A-Z])(?=[A-Za-z0-9_./+=:-]*[a-z])(?=[A-Za-z0-9_./+=:-]*\d)[A-Za-z0-9_./+=:-]{24,}\b"), "[REDACTED TOKEN]"),
    (re.compile(r"(?<!\w)\+?\d[\d\s().-]{7,}\d(?!\w)"), "[REDACTED NUMBER]"),
)
_IPV6_CANDIDATE_RE = re.compile(r"(?<![0-9A-Fa-f:.])(?=[0-9A-Fa-f:.]{0,38}:)[0-9A-Fa-f:.]{2,39}(?![0-9A-Fa-f:.])")


def redact_text(text: str) -> str:
    """Mask common secrets before any excerpt can reach UI or report output."""
    safe = text
    for pattern, replacement in _REDACTION_PATTERNS:
        safe = pattern.sub(replacement, safe)
    def mask_ipv6(match: re.Match) -> str:
        try:
            ipaddress.IPv6Address(match.group())
        except ipaddress.AddressValueError:
            return match.group()
        return "[REDACTED IP]"
    safe = _IPV6_CANDIDATE_RE.sub(mask_ipv6, safe)
    return safe


def safe_evidence(text: str, start: int, end: int, *, limit: int = MAX_EVIDENCE_CHARS) -> str:
    if not text or limit < 16:
        return "Matched text omitted."
    pad = max(8, (limit - max(0, end - start)) // 2)
    left, right = max(0, start - pad), min(len(text), end + pad)
    excerpt = redact_text(text[left:right]).replace("\r", " ").replace("\n", " ")
    excerpt = re.sub(r"[\x00-\x1f\x7f]", " ", excerpt)
    prefix = "…" if left else ""
    suffix = "…" if right < len(text) else ""
    available = max(0, limit - len(prefix) - len(suffix))
    return prefix + excerpt[:available].rstrip() + suffix


def make_finding(*, text: str, start: int, end: int, category: str, rule_id: str,
                 severity: Severity, detector: str, strength: str,
                 explanation: str, remediation: str, source: str = "Input") -> Finding:
    return Finding(
        category=category,
        rule_id=rule_id,
        severity=severity,
        detector=detector,
        strength=strength,
        explanation=explanation,
        evidence=safe_evidence(text, start, end),
        remediation=remediation,
        source=source,
    )
