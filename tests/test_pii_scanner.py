import pytest

from src.security.pii_scanner import luhn_valid, mask_value, scan_pii
from src.security.schemas import ScanError, redact_text


def test_email_phone_and_ssn_like_findings_are_masked():
    text = "Contact alice@example.com or +1 (415) 555-0198. Ref 123-45-6789."
    findings = scan_pii(text)
    assert {item.category for item in findings} == {"email", "phone", "government_id"}
    combined = " ".join(item.evidence for item in findings)
    for raw in ("alice@example.com", "415) 555-0198", "123-45-6789"):
        assert raw not in combined
    assert any("US SSN-like" in item.explanation for item in findings)


def test_luhn_check_detects_valid_card_candidate_and_masks_it():
    assert luhn_valid("4111 1111 1111 1111")
    assert not luhn_valid("4111 1111 1111 1112")
    findings = scan_pii("card=4111 1111 1111 1111")
    card = next(item for item in findings if item.category == "payment_card")
    assert "4111 1111 1111 1111" not in card.evidence
    assert card.severity.value == "High"


@pytest.mark.parametrize("text,category", [
    ("api_key=sk_live_1234567890abcdef", "api_key"),
    ("password: dont-print-this-value", "password"),
    ("-----BEGIN PRIVATE KEY-----secret-content-----END PRIVATE KEY-----", "private_key"),
])
def test_credential_shapes_are_never_echoed(text, category):
    findings = scan_pii(text)
    found = next(item for item in findings if item.category == category)
    assert "secret-content" not in found.evidence
    assert "dont-print-this-value" not in found.evidence
    assert "sk_live_1234567890abcdef" not in found.evidence


def test_ip_address_is_validated_and_invalid_ipv4_is_ignored():
    findings = scan_pii("host 192.168.1.12, fd00::1 and invalid 999.1.1.1")
    ips = [item for item in findings if item.category == "ip_address"]
    assert len(ips) == 2
    assert all("192.168.1.12" not in item.evidence and "fd00::1" not in item.evidence for item in ips)


def test_invalid_ssn_like_and_non_luhn_card_are_not_reported():
    findings = scan_pii("000-12-3456 4111 1111 1111 1112")
    assert not any(item.category in ("government_id", "payment_card") for item in findings)


def test_empty_input_and_category_selection():
    assert scan_pii("") == []
    assert [item.category for item in scan_pii("a@example.com 4155550198", enabled_categories=["email"])] == ["email"]
    with pytest.raises(ScanError):
        scan_pii("safe", enabled_categories=["unknown"])


def test_oversized_and_invalid_text_are_rejected():
    with pytest.raises(ScanError):
        scan_pii("x" * 20_001)
    with pytest.raises(ScanError):
        scan_pii(None)  # type: ignore[arg-type]


def test_mask_value_never_returns_the_complete_sensitive_value():
    assert mask_value("alice@example.com", "email") != "alice@example.com"
    assert mask_value("4111111111111111", "payment_card").endswith("1111")


def test_configured_sensitive_labels_omit_assigned_values():
    finding = scan_pii("employee_id=EMP-938421", enabled_categories=["custom_identifier"],
                       custom_labels=["employee_id"])[0]
    assert finding.category == "custom_identifier"
    assert "EMP-938421" not in finding.evidence
    with pytest.raises(ScanError):
        scan_pii("safe", custom_labels=[".*"])


def test_evidence_redaction_masks_natural_language_secret_assignments():
    raw = "the secret is AlphaBeta1234567890-Gamma and should remain private"
    safe = redact_text(raw)
    assert "AlphaBeta1234567890-Gamma" not in safe
    assert "[REDACTED CREDENTIAL]" in safe
