"""Conservative local text redaction for the firewall's output path."""
from __future__ import annotations

import ipaddress
import re

from src.security.pii_scanner import luhn_valid
from src.security.schemas import redact_text

_EMAIL = re.compile(r"(?<![\w.+-])[A-Z0-9._%+-]{1,64}@[A-Z0-9.-]{1,190}\.[A-Z]{2,24}(?![\w.-])", re.I)
_SSN = re.compile(r"(?<!\d)(?!000|666|9\d\d)\d{3}[- ]?(?!00)\d{2}[- ]?(?!0000)\d{4}(?!\d)")
_CARD = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")
_PHONE = re.compile(r"(?<!\w)\+?\d[\d ().-]{7,18}\d(?!\w)")
_IPV4 = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")


def redact_sensitive_text(text: str) -> str:
    """Redact common PII and credentials. Rules are conservative and may redact benign numbers."""
    result = redact_text(text)
    result = _EMAIL.sub("[REDACTED EMAIL]", result)
    result = _SSN.sub("[REDACTED GOVERNMENT ID]", result)
    result = _CARD.sub(lambda m: "[REDACTED PAYMENT CARD]" if luhn_valid(m.group()) else m.group(), result)
    result = _PHONE.sub(lambda m: "[REDACTED PHONE]" if 7 <= sum(c.isdigit() for c in m.group()) <= 15 else m.group(), result)

    def safe_ipv4(match: re.Match) -> str:
        try:
            ipaddress.IPv4Address(match.group())
        except ipaddress.AddressValueError:
            return match.group()
        return "[REDACTED IP]"

    return _IPV4.sub(safe_ipv4, result)


def redact_phrases(text: str, phrases: tuple[str, ...], *, case_sensitive: bool = False) -> str:
    """Replace bounded literal custom-rule phrases without interpreting them as regex."""
    result = text
    for phrase in sorted(phrases, key=len, reverse=True):
        if not phrase:
            continue
        result = re.sub(re.escape(phrase), "[REDACTED POLICY MATCH]", result,
                        flags=0 if case_sensitive else re.IGNORECASE)
    return result
