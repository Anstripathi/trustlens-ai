"""Heuristic checks for prompt-injection-style instruction patterns."""
from __future__ import annotations

import re

from .schemas import Finding, Severity, make_finding, validate_text

_RULES = (
    ("override_instructions", re.compile(r"\b(?:ignore|disregard|forget|override)\b.{0,100}\b(?:previous|prior|above|earlier|all|system|developer)\b.{0,40}\b(?:instructions?|prompts?|messages?|rules?)\b", re.I | re.S), Severity.HIGH,
     "The text asks to replace or ignore earlier instructions.", "Keep trusted instructions separate from untrusted user content.", "High"),
    ("extract_hidden_prompt", re.compile(r"\b(?:reveal|show|print|repeat|dump|disclose|quote|provide)\b.{0,80}\b(?:hidden|system|developer|internal)\b.{0,35}\b(?:prompts?|instructions?|messages?)\b", re.I | re.S), Severity.HIGH,
     "The text requests hidden or internal instructions.", "Do not expose system prompts, credentials, or internal configuration.", "High"),
    ("hierarchy_manipulation", re.compile(r"\b(?:treat|consider|obey|follow)\b.{0,70}\b(?:this|my|next|following)\b.{0,40}\b(?:system|developer|highest.priority|root)\b.{0,30}\b(?:instruction|message|prompt)\b", re.I | re.S), Severity.HIGH,
     "The text attempts to assign higher authority to user-provided content.", "Enforce instruction hierarchy outside the user-controlled prompt.", "Medium"),
    ("role_manipulation", re.compile(r"\b(?:you are now|act as|role[- ]?play as|pretend to be)\b.{0,70}\b(?:admin|developer|system|unrestricted|root|different assistant)\b", re.I | re.S), Severity.MEDIUM,
     "The text tries to change the assistant's role or privilege level.", "Do not grant privileges based on role claims in untrusted input.", "Medium"),
    ("restriction_bypass", re.compile(r"\b(?:bypass|disable|evade|circumvent|ignore|override)\b.{0,50}\b(?:safety|security|guardrails?|restrictions?|filters?|polic(?:y|ies))\b", re.I | re.S), Severity.HIGH,
     "The text requests bypassing a safety or security restriction.", "Keep safety checks enabled and treat the request as untrusted content.", "High"),
    ("confidential_disclosure", re.compile(r"\b(?:reveal|send|print|exfiltrate|show|disclose|return)\b.{0,60}\b(?:passwords?|secrets?|api keys?|credentials?|private keys?|confidential data|customer records?)\b", re.I | re.S), Severity.HIGH,
     "The text requests disclosure of a credential or confidential data.", "Do not disclose secrets or personal data; inspect data access and output controls.", "High"),
    ("tool_manipulation", re.compile(r"\b(?:use|call|invoke|run)\b.{0,45}\b(?:browser|python|shell|tool|function|connector)\b.{0,70}\b(?:send|upload|exfiltrate|leak|read|disclose|post)\b", re.I | re.S), Severity.HIGH,
     "The text combines a tool request with a data transfer or disclosure instruction.", "Require explicit authorization and narrow tool permissions; do not execute scanned text.", "Medium"),
    ("encoded_instruction", re.compile(r"\b(?:decode|base64|hex|rot13|unicode)\b.{0,100}\b(?:instruction|prompt|ignore|reveal|bypass|secret)\b", re.I | re.S), Severity.MEDIUM,
     "The text couples decoding or obfuscation with a potentially sensitive instruction.", "Inspect decoded content in a safe, isolated review flow; this scanner does not execute it.", "Low"),
    ("encoded_payload_marker", re.compile(r"(?i)(?:\bbase64\b.{0,30})?(?:[A-Za-z0-9+/]{48,}={0,2})"), Severity.LOW,
     "A long base64-like payload was found; encoded content may be benign and is not decoded here.", "Review the payload only if relevant; encoded text alone does not establish malicious intent.", "Low"),
)
_INVISIBLE_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060\ufeff]")


def scan_prompt_injection(text: str) -> list[Finding]:
    """Return heuristic findings; this does not establish that a prompt is malicious."""
    text = validate_text(text)
    findings = []
    for rule_id, pattern, severity, explanation, remediation, strength in _RULES:
        for match in pattern.finditer(text):
            findings.append(make_finding(
                text=text, start=match.start(), end=match.end(), category="prompt_injection",
                rule_id=f"injection.{rule_id}", severity=severity,
                detector="Local heuristic pattern", strength=strength,
                explanation=explanation, remediation=remediation,
            ))
    for match in _INVISIBLE_RE.finditer(text):
        findings.append(make_finding(
            text=text, start=match.start(), end=match.end(), category="prompt_injection",
            rule_id="injection.invisible_control", severity=Severity.LOW,
            detector="Deterministic character check", strength="Low",
            explanation="An invisible Unicode formatting character may obscure displayed text.",
            remediation="Normalize or inspect invisible formatting characters before interpreting the text.",
        ))
    return findings
