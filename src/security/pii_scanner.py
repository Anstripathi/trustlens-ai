"""Bounded local scanner for common PII and credential-like values."""
from __future__ import annotations

import ipaddress
import re

from .schemas import Finding, ScanError, Severity, make_finding, validate_text

PII_CATEGORIES = (
    "email", "phone", "government_id", "payment_card", "api_key",
    "password", "private_key", "ip_address", "custom_identifier",
)

EMAIL_RE = re.compile(r"(?<![\w.+-])[A-Z0-9._%+-]{1,64}@[A-Z0-9.-]{1,190}\.[A-Z]{2,24}(?![\w.-])", re.I)
SSN_RE = re.compile(r"(?<!\d)(?!000|666|9\d\d)\d{3}[- ]?(?!00)\d{2}[- ]?(?!0000)\d{4}(?!\d)")
CARD_RE = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")
PHONE_RE = re.compile(r"(?<!\w)\+?\d[\d ().-]{7,18}\d(?!\w)")
IPV4_RE = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
IPV6_CANDIDATE_RE = re.compile(r"(?<![0-9A-Fa-f:.])[0-9A-Fa-f:.]{2,39}(?![0-9A-Fa-f:.])")
API_RE = re.compile(r"(?i)\b(?:AKIA[0-9A-Z]{16}|(?:sk|pk)_(?:live|test)_[A-Za-z0-9]{12,}|gh[pousr]_[A-Za-z0-9]{20,}|xox[baprs]-[A-Za-z0-9-]{12,})\b")
ASSIGNED_SECRET_RE = re.compile(r"(?i)\b(?:api[_ -]?key|access[_ -]?token|auth[_ -]?token|secret|token)\s*(?:[:=]|\bis\b)\s*['\"]?([A-Za-z0-9_./+=:-]{6,160})")
PASSWORD_RE = re.compile(r"(?i)\b(?:password|passwd|pwd)\s*(?:[:=]|\bis\b)\s*['\"]?([^\s,;\"']{1,120})")
PRIVATE_KEY_RE = re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----[\s\S]{0,6000}(?:-----END (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----)?", re.I)


def luhn_valid(value: str) -> bool:
    digits = [int(char) for char in value if char.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    parity = len(digits) % 2
    for index, digit in enumerate(digits):
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def _overlaps(start: int, end: int, spans: list[tuple[int, int]]) -> bool:
    return any(start < other_end and end > other_start for other_start, other_end in spans)


def mask_value(value: str, category: str) -> str:
    """Return a deliberately partial display value; never return the full match."""
    if category == "email" and "@" in value:
        local, domain = value.split("@", 1)
        domain_parts = domain.split(".")
        safe_local = (local[:1] + "***") if local else "***"
        safe_domain = (domain_parts[0][:1] + "***") if domain_parts and domain_parts[0] else "***"
        suffix = "." + domain_parts[-1] if len(domain_parts) > 1 else ""
        return f"{safe_local}@{safe_domain}{suffix}"
    if category == "payment_card":
        digits = "".join(char for char in value if char.isdigit())
        return "•••• •••• •••• " + digits[-4:]
    if category == "phone":
        digits = "".join(char for char in value if char.isdigit())
        return "[REDACTED PHONE] ending " + digits[-2:]
    return f"[REDACTED {category.upper().replace('_', ' ')}]"


def scan_pii(text: str, *, enabled_categories: tuple[str, ...] | list[str] | None = None,
             custom_labels: tuple[str, ...] | list[str] = ()) -> list[Finding]:
    """Scan text locally and return masked evidence, never the complete sensitive match."""
    text = validate_text(text)
    enabled = set(PII_CATEGORIES if enabled_categories is None else enabled_categories)
    unknown = enabled - set(PII_CATEGORIES)
    if unknown:
        raise ScanError("Unknown sensitive-data category selected.")
    if not isinstance(custom_labels, (tuple, list)) or any(not isinstance(label, str) for label in custom_labels):
        raise ScanError("Custom sensitive field labels must be provided as text values.")
    labels = list(dict.fromkeys(custom_labels))
    if len(labels) > 6 or any(not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,31}", label) for label in labels):
        raise ScanError("Custom field labels must be up to six names of 1–32 letters, digits, underscores, or hyphens.")

    findings: list[Finding] = []
    occupied: list[tuple[int, int]] = []

    def add(match: re.Match, category: str, severity: Severity, rule: str,
            why: str, remediation: str, *, strength: str = "High") -> None:
        findings.append(make_finding(
            text=text, start=match.start(), end=match.end(), category=category,
            rule_id=rule, severity=severity, detector="Deterministic pattern/validation rule",
            strength=strength, explanation=why, remediation=remediation,
        ))
        occupied.append((match.start(), match.end()))

    if "private_key" in enabled:
        for match in PRIVATE_KEY_RE.finditer(text):
            add(match, "private_key", Severity.CRITICAL, "pii.private_key_marker",
                "A private-key marker was found. The scanner does not validate the key or its ownership.",
                "Remove the key from the content and rotate it if it may have been exposed.")

    if "email" in enabled:
        for match in EMAIL_RE.finditer(text):
            add(match, "email", Severity.LOW, "pii.email",
                "Text matches a common email-address structure; validity is not verified.",
                "Remove or redact the address unless it is required for the assessment.", strength="Medium")

    if "government_id" in enabled:
        for match in SSN_RE.finditer(text):
            add(match, "government_id", Severity.HIGH, "pii.us_ssn_like",
                "Text matches a US SSN-like numeric pattern. No identity or jurisdictional validity is verified.",
                "Redact the identifier and review the source and applicable jurisdictional requirements.", strength="Medium")

    if "payment_card" in enabled:
        for match in CARD_RE.finditer(text):
            if luhn_valid(match.group()) and not _overlaps(match.start(), match.end(), occupied):
                add(match, "payment_card", Severity.HIGH, "pii.payment_card_luhn",
                    "A 13–19 digit value passes the Luhn checksum; this does not prove it is an active card.",
                    "Remove the value and use a payment provider's tokenization process instead.")

    if "api_key" in enabled:
        for match in API_RE.finditer(text):
            if not _overlaps(match.start(), match.end(), occupied):
                add(match, "api_key", Severity.HIGH, "pii.known_token_shape",
                    "Text matches a recognized API-token shape; the token is not validated.",
                    "Revoke and rotate the credential if genuine, then remove it from the source.")
        for match in ASSIGNED_SECRET_RE.finditer(text):
            start, end = match.span(1)
            if not _overlaps(start, end, occupied):
                token_match = re.match(r"[\s\S]+", text[start:end])
                add(_SpanMatch(token_match, start), "api_key", Severity.HIGH, "pii.assigned_secret",
                    "A value is assigned to a token, secret, or API-key field; the value is not validated.",
                    "Remove the value and rotate the credential if it may be genuine.", strength="Medium")

    if "password" in enabled:
        for match in PASSWORD_RE.finditer(text):
            start, end = match.span(1)
            if not _overlaps(start, end, occupied):
                token_match = re.match(r"[\s\S]+", text[start:end])
                add(_SpanMatch(token_match, start), "password", Severity.HIGH, "pii.password_assignment",
                    "A value appears assigned to a password-like field; the value is not validated.",
                    "Remove it and reset the credential if it may be genuine.", strength="Medium")

    # Identify IPv4 literals before the broader phone-like rule so dotted
    # octets are not consumed as a phone match.
    if "ip_address" in enabled:
        for match in IPV4_RE.finditer(text):
            if _overlaps(match.start(), match.end(), occupied):
                continue
            try:
                ipaddress.IPv4Address(match.group())
            except ipaddress.AddressValueError:
                continue
            add(match, "ip_address", Severity.LOW, "pii.ipv4_address",
                "Text is a syntactically valid IPv4 address; whether it is sensitive depends on context.",
                "Redact internal or identifying addresses when they are not needed.", strength="Medium")
        for match in IPV6_CANDIDATE_RE.finditer(text):
            if _overlaps(match.start(), match.end(), occupied) or ":" not in match.group():
                continue
            try:
                ipaddress.IPv6Address(match.group())
            except ipaddress.AddressValueError:
                continue
            add(match, "ip_address", Severity.LOW, "pii.ipv6_address",
                "Text is a syntactically valid IPv6 address; whether it is sensitive depends on context.",
                "Redact internal or identifying addresses when they are not needed.", strength="Medium")

    if "custom_identifier" in enabled and labels:
        label_pattern = re.compile(
            r"(?i)(?<!\w)(?:" + "|".join(re.escape(label) for label in labels)
            + r")\s*[:=]\s*([A-Za-z0-9][A-Za-z0-9_-]{2,63})"
        )
        for match in label_pattern.finditer(text):
            start, end = match.span(1)
            if _overlaps(start, end, occupied):
                continue
            findings.append(Finding(
                category="custom_identifier", rule_id="pii.configured_field_label",
                severity=Severity.MEDIUM, detector="Configured field-label rule",
                strength="Medium", explanation="A value is assigned to a user-configured sensitive field label.",
                evidence="Configured sensitive field matched; value omitted.",
                remediation="Redact this field or remove the value before sharing the content.",
            ))
            occupied.append((start, end))

    if "phone" in enabled:
        for match in PHONE_RE.finditer(text):
            digits = sum(char.isdigit() for char in match.group())
            if 7 <= digits <= 15 and not _overlaps(match.start(), match.end(), occupied):
                add(match, "phone", Severity.MEDIUM, "pii.phone_like",
                    "Text resembles a phone number; locale, assignment, and validity are not verified.",
                    "Redact the number unless it is necessary for the assessment.", strength="Medium")

    return findings


class _SpanMatch:
    """Small adapter so captured credential spans use the shared safe evidence builder."""
    def __init__(self, match, offset: int):
        self._match, self._offset = match, offset

    def start(self):
        return self._offset + self._match.start()

    def end(self):
        return self._offset + self._match.end()
