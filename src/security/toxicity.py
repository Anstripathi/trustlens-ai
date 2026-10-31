"""Small lexical unsafe-output screen; not a semantic moderation model."""
from __future__ import annotations

import re

from .schemas import Finding, ScanError, Severity, validate_text

TOXICITY_CATEGORIES = (
    "abusive_language", "harassment", "threats", "self_harm",
    "sexual_exploitation", "violence", "hate",
)

_RULES = (
    ("abusive_language", "toxicity.directed_insult", re.compile(r"\b(?:you are|you're)\s+(?:a\s+)?(?:idiot|moron|worthless|pathetic)\b", re.I), Severity.MEDIUM,
     "Directed-insult phrase matched a local rule.", "Review the surrounding context and apply an appropriate respectful-language policy."),
    ("harassment", "toxicity.targeted_harassment", re.compile(r"\b(?:i|we)\s+(?:will|are going to)\s+(?:keep\s+)?(?:following|messaging|calling|targeting)\s+you\b", re.I), Severity.HIGH,
     "A targeted repeated-contact or targeting phrase matched a local rule.", "Review for targeted abuse and use a human review process for consequential decisions."),
    ("threats", "toxicity.direct_threat", re.compile(r"\b(?:i|we)\s+(?:will|am going to|are going to)\s+(?:hurt|attack|kill|burn)\s+you\b", re.I), Severity.HIGH,
     "A direct threat-shaped phrase matched a local rule.", "Use established safety procedures and qualified human review where appropriate."),
    ("self_harm", "toxicity.self_harm_encouragement", re.compile(r"\b(?:kill yourself|hurt yourself|you should die|go kill yourself)\b", re.I), Severity.CRITICAL,
     "A self-harm encouragement phrase matched a local rule.", "Escalate according to the relevant safety process; do not rely on this lexical match alone."),
    ("self_harm", "toxicity.self_harm_disclosure", re.compile(r"\b(?:i|we)\s+(?:want|plan|intend)\s+to\s+(?:kill myself|hurt myself|end my life)\b", re.I), Severity.HIGH,
     "A first-person self-harm disclosure phrase matched a local rule.", "Use a qualified, supportive review process; the detector cannot assess intent or immediacy."),
    ("sexual_exploitation", "toxicity.minor_sexual_content", re.compile(r"\b(?:sexual|explicit|nude)\s+(?:images?|content|activity)\s+(?:of|involving)\s+(?:a child|children|a minor|minors|underage people)\b", re.I), Severity.CRITICAL,
     "A phrase combining sexual content and minors matched a local rule.", "Do not reproduce or circulate the content; follow the applicable safeguarding process."),
    ("sexual_exploitation", "toxicity.minor_sexual_content_reverse", re.compile(r"\b(?:a child|children|a minor|minors|underage people)\b.{0,35}\b(?:sexual|explicit|nude)\s+(?:images?|content|activity)\b", re.I), Severity.CRITICAL,
     "A phrase combining minors and sexual content matched a local rule.", "Do not reproduce or circulate the content; follow the applicable safeguarding process."),
    ("violence", "toxicity.weapon_instructions_request", re.compile(r"\b(?:instructions?|steps?|guide)\s+(?:to|for)\s+(?:make|build|assemble)\s+(?:a\s+)?(?:bomb|weapon)\b", re.I), Severity.HIGH,
     "A request for weapon-construction instructions matched a local rule.", "Review against the applicable content-safety policy; this detector does not assess context."),
    ("hate", "toxicity.group_harm_advocacy", re.compile(r"\b(?:all|every)\s+(?:members|people)\s+of\s+(?:that|the)\s+(?:race|religion|ethnicity|nationality|group)\s+(?:should|deserve to)\s+(?:die|be harmed|be removed)\b", re.I), Severity.HIGH,
     "A group-directed harm phrase matched a local rule.", "Review for hateful or discriminatory content with appropriate human context."),
)


def scan_unsafe_output(text: str, *, enabled_categories: tuple[str, ...] | list[str] | None = None) -> list[Finding]:
    """Scan locally and deliberately omit matched harmful text from evidence."""
    text = validate_text(text)
    enabled = set(TOXICITY_CATEGORIES if enabled_categories is None else enabled_categories)
    if enabled - set(TOXICITY_CATEGORIES):
        raise ScanError("Unknown unsafe-content category selected.")
    findings: list[Finding] = []
    for category, rule_id, pattern, severity, explanation, remediation in _RULES:
        if category not in enabled:
            continue
        for _match in pattern.finditer(text):
            findings.append(Finding(
                category=category,
                rule_id=rule_id,
                severity=severity,
                detector="Local lexical heuristic",
                strength="Heuristic; context not evaluated",
                explanation=explanation + " This is not a semantic verdict.",
                evidence=f"Matched rule {rule_id}; source text omitted.",
                remediation=remediation,
            ))
    return findings
