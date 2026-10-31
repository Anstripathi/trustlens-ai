import pytest

from src.security.jailbreak import JAILBREAK_CATEGORIES, run_jailbreak_suite
from src.security.schemas import ScanError


def test_suite_covers_categories_deterministically_and_flags_cases():
    first = run_jailbreak_suite()
    second = run_jailbreak_suite()
    assert first == second
    assert {row["category"] for row in first} == set(JAILBREAK_CATEGORIES)
    assert all(row["flagged"] and row["rule_ids"] for row in first)


def test_suite_runs_selected_category_and_has_bounded_case_count():
    cases = run_jailbreak_suite(["direct_override"], max_cases=1)
    assert len(cases) == 1 and cases[0]["category"] == "direct_override"
    with pytest.raises(ScanError):
        run_jailbreak_suite(["direct_override"], max_cases=25)
    with pytest.raises(ScanError):
        run_jailbreak_suite(["not-a-category"])
