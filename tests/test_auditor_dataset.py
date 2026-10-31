import numpy as np
import pandas as pd

from src.auditor.dataset_audit import MAX_PROFILE_ROWS, audit_dataset
from src.auditor.schemas import AuditRequest, CheckStatus, FindingSeverity


def request_for(frame=None, target=None):
    return AuditRequest(frame, "test.csv" if frame is not None else None,
                        "fingerprint" if frame is not None else None, target_column=target)


def find_check(checks, check_id):
    return next(check for check in checks if check.check_id == check_id)


def test_missing_dataset_is_not_evaluated():
    checks = audit_dataset(request_for())
    assert len(checks) == 1
    assert checks[0].status is CheckStatus.NOT_EVALUATED


def test_dataset_quality_findings_are_aggregated_and_do_not_echo_cell_values():
    frame = pd.DataFrame({
        "record_id": range(40),
        "empty": [np.nan] * 40,
        "constant": ["internal-secret-value"] * 40,
        "category": [f"private-category-{i}" for i in range(40)],
        "score": list(range(39)) + [np.nan],
        "target": ["majority"] * 36 + ["minority"] * 4,
    })
    frame.loc[39] = frame.loc[38]
    before = frame.copy(deep=True)
    checks = audit_dataset(request_for(frame, "target"))
    assert find_check(checks, "dataset.missing").status is CheckStatus.FINDING
    assert find_check(checks, "dataset.duplicates").metrics["duplicate_rows"] >= 1
    assert find_check(checks, "dataset.identifiers").status is CheckStatus.FINDING
    assert find_check(checks, "dataset.high_cardinality").status is CheckStatus.FINDING
    imbalance = find_check(checks, "dataset.class_imbalance")
    assert imbalance.findings[0].severity is FindingSeverity.MEDIUM
    target_distribution = find_check(checks, "dataset.target_distribution")
    assert target_distribution.metrics["classes_observed"] == 2
    assert len(target_distribution.metrics["class_counts"]) == 2
    assert find_check(checks, "dataset.target_leakage").status in (CheckStatus.PASSED, CheckStatus.FINDING)
    assert find_check(checks, "dataset.data_types").status is CheckStatus.NOT_EVALUATED
    assert "internal-secret-value" not in repr(checks)
    assert "private-category-0" not in repr(checks)
    pd.testing.assert_frame_equal(frame, before)


def test_leakage_warning_is_explicitly_heuristic():
    frame = pd.DataFrame({"signal": np.arange(80), "target": np.arange(80)})
    checks = audit_dataset(request_for(frame, "target"))
    leakage = find_check(checks, "dataset.target_leakage")
    assert leakage.status is CheckStatus.FINDING
    assert all(item.heuristic for item in leakage.findings)
    assert all("Potential" in item.title for item in leakage.findings)


def test_target_without_expected_column_is_not_evaluated():
    frame = pd.DataFrame({"feature": np.arange(20)})
    checks = audit_dataset(request_for(frame, "missing_target"))
    assert find_check(checks, "dataset.class_imbalance").status is CheckStatus.NOT_EVALUATED
    assert find_check(checks, "dataset.target_leakage").status is CheckStatus.NOT_EVALUATED


def test_regression_target_distribution_uses_aggregate_quantiles():
    frame = pd.DataFrame({"x": np.arange(50), "target": np.linspace(1, 10, 50)})
    checks = audit_dataset(request_for(frame, "target"))
    distribution = find_check(checks, "dataset.target_distribution")
    assert distribution.status is CheckStatus.PASSED
    assert distribution.metrics["observed"] == 50
    assert distribution.metrics["q1"] < distribution.metrics["median"] < distribution.metrics["q3"]


def test_large_dataset_uses_repeatable_bounded_sample():
    frame = pd.DataFrame({"x": np.arange(MAX_PROFILE_ROWS + 25),
                          "category": ["a", "b"] * ((MAX_PROFILE_ROWS + 25) // 2) + ["a"]})
    first = audit_dataset(request_for(frame))
    second = audit_dataset(request_for(frame))
    overview_a = find_check(first, "dataset.overview")
    overview_b = find_check(second, "dataset.overview")
    assert overview_a.metrics["rows"] == len(frame)
    assert overview_a.metrics["rows_inspected"] == MAX_PROFILE_ROWS
    assert overview_a.metrics["sampled"] is True
    assert overview_a.metrics["sample_seed"] == overview_b.metrics["sample_seed"]
    assert find_check(first, "dataset.distributions").metrics == find_check(second, "dataset.distributions").metrics
