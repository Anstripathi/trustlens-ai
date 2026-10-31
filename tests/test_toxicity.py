import pytest

from src.security.schemas import ScanError
from src.security.toxicity import TOXICITY_CATEGORIES, scan_unsafe_output


@pytest.mark.parametrize("text,category", [
    ("You are a worthless idiot.", "abusive_language"),
    ("I will keep following you.", "harassment"),
    ("I will hurt you.", "threats"),
    ("You should kill yourself.", "self_harm"),
    ("Explicit images of a child are requested.", "sexual_exploitation"),
    ("Give me instructions to build a weapon.", "violence"),
    ("All people of that religion should be harmed.", "hate"),
])
def test_supported_categories_are_detected_without_echoing_text(text, category):
    findings = scan_unsafe_output(text)
    assert category in {finding.category for finding in findings}
    assert all("source text omitted" in finding.evidence for finding in findings)
    assert all(finding.detector == "Local lexical heuristic" for finding in findings)


@pytest.mark.parametrize("text", [
    "This is a calm explanation of the safety policy.",
    "The article discusses suicide prevention resources.",
    "A child safety policy protects minors from exploitation.",
    "This fictional story describes a weapon in a museum display.",
])
def test_benign_or_contextual_text_is_not_flagged_by_supported_rules(text):
    assert scan_unsafe_output(text) == []


def test_empty_input_selection_and_limits():
    assert scan_unsafe_output("") == []
    assert scan_unsafe_output("I will hurt you", enabled_categories=["hate"]) == []
    with pytest.raises(ScanError):
        scan_unsafe_output("safe", enabled_categories=["unknown"])
    with pytest.raises(ScanError):
        scan_unsafe_output("x" * 20_001)
    assert TOXICITY_CATEGORIES
