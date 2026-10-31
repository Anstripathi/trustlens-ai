"""Bounded, read-only dataset audit built on TrustLens' existing profiler."""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from .. import config
from ..data_profile import correlation_matrix, profile
from ..task_detection import infer_task
from .schemas import (
    AuditCategory,
    AuditCheck,
    AuditFinding,
    AuditRequest,
    CheckStatus,
    FindingSeverity,
)

MAX_PROFILE_ROWS = 10_000
MAX_PROFILE_COLUMNS = 30
MAX_CORRELATION_PAIRS = 50
PROFILE_SEED = 1729


def _check(check_id: str, status: CheckStatus, summary: str, *, findings=(), metrics=None,
           limitations=()) -> AuditCheck:
    return AuditCheck(
        check_id=check_id,
        category=AuditCategory.DATASET,
        status=status,
        summary=summary,
        findings=tuple(findings),
        metrics=dict(metrics or {}),
        limitations=tuple(limitations),
    )


def _finding(check_id: str, suffix: str, severity: FindingSeverity, title: str,
             explanation: str, evidence: dict, *, heuristic: bool = False) -> AuditFinding:
    return AuditFinding(
        finding_id=f"{check_id}.{suffix}",
        category=AuditCategory.DATASET,
        severity=severity,
        title=title,
        explanation=explanation,
        evidence=evidence,
        source_check=check_id,
        heuristic=heuristic,
    )


def _status(findings: list[AuditFinding]) -> CheckStatus:
    return CheckStatus.FINDING if findings else CheckStatus.PASSED


def _sample_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    if len(df) <= MAX_PROFILE_ROWS:
        return df, False
    return df.sample(n=MAX_PROFILE_ROWS, random_state=PROFILE_SEED), True


def _safe_column_summary(frame: pd.DataFrame, p: dict) -> dict:
    """Return aggregate-only distributions; never include categorical values."""
    numeric = frame.select_dtypes(include="number")
    numeric_summary: dict[str, dict[str, float | int]] = {}
    for name in list(numeric.columns)[:MAX_PROFILE_COLUMNS]:
        values = numeric[name].replace([np.inf, -np.inf], np.nan).dropna()
        if values.empty:
            continue
        quantiles = values.quantile([0.25, 0.5, 0.75])
        numeric_summary[str(name)] = {
            "observed": int(values.size),
            "q1": float(quantiles.loc[0.25]),
            "median": float(quantiles.loc[0.5]),
            "q3": float(quantiles.loc[0.75]),
        }
    categorical_summary: dict[str, dict[str, int]] = {}
    cat = p["categorical_summary"]
    for name in list(cat.index)[:MAX_PROFILE_COLUMNS]:
        categorical_summary[str(name)] = {
            "unique_values": int(cat.loc[name, "unique"]),
            "most_common_count": int(cat.loc[name, "top_freq"]),
        }

    correlations: list[dict[str, float | str]] = []
    corr = correlation_matrix(frame, max_cols=MAX_PROFILE_COLUMNS)
    if corr is not None:
        cols = list(corr.columns)
        for i, left in enumerate(cols):
            for right in cols[i + 1:]:
                value = corr.loc[left, right]
                if pd.notna(value):
                    correlations.append({"feature_a": str(left), "feature_b": str(right),
                                         "correlation": float(value)})
        correlations.sort(key=lambda pair: abs(float(pair["correlation"])), reverse=True)
    return {
        "numeric": numeric_summary,
        "categorical": categorical_summary,
        "strongest_numeric_correlations": correlations[:MAX_CORRELATION_PAIRS],
    }


def audit_dataset(request: AuditRequest) -> tuple[AuditCheck, ...]:
    """Inspect dataset quality and heuristic risks without modifying the input."""
    df = request.dataset
    if not isinstance(df, pd.DataFrame):
        return (_check("dataset.overview", CheckStatus.NOT_EVALUATED,
                       "No dataset is available for assessment."),)
    if df.empty or df.shape[1] == 0:
        return (_check("dataset.overview", CheckStatus.FAILED,
                       "The active dataset has no rows or columns."),)

    sampled, is_sampled = _sample_frame(df)
    row_scope = {
        "rows_inspected": int(len(sampled)),
        "dataset_rows": int(len(df)),
        "sampled": is_sampled,
        "sample_seed": PROFILE_SEED if is_sampled else None,
    }
    target = request.target_column if request.target_column in sampled.columns else None
    p = profile(sampled, target)
    limitations = ((f"Quality statistics are estimated from a deterministic sample of {len(sampled):,} rows; "
                    "full dataset dimensions and dtypes are exact.",) if is_sampled else ())

    checks: list[AuditCheck] = [
        _check("dataset.overview", CheckStatus.PASSED, "Dataset dimensions are available.",
               metrics={"rows": int(len(df)), "columns": int(df.shape[1]), **row_scope},
               limitations=limitations),
    ]

    missing_findings: list[AuditFinding] = []
    for name, row in p["missing"].iterrows():
        count = int(row["missing"])
        rate = float(row["missing_%"]) / 100.0
        if count:
            severity = (FindingSeverity.HIGH if rate >= 0.20 else
                        FindingSeverity.MEDIUM if rate >= 0.05 else FindingSeverity.LOW)
            missing_findings.append(_finding(
                "dataset.missing", str(name), severity, "Missing values observed",
                "Missingness may affect feature coverage or target availability; review the intended handling.",
                {"column": str(name), "missing_count": count, "missing_rate": rate, **row_scope},
            ))
    checks.append(_check("dataset.missing", _status(missing_findings),
                         "Missing-value rates were inspected.", findings=missing_findings,
                         metrics={"columns_with_missing": len(missing_findings)}, limitations=limitations))

    duplicate_count = int(p["duplicate_rows"])
    duplicate_findings = []
    if duplicate_count:
        duplicate_findings.append(_finding(
            "dataset.duplicates", "duplicate_rows", FindingSeverity.MEDIUM,
            "Duplicate rows observed",
            "Repeated rows may be valid observations or may affect evaluation; confirm their provenance before removing them.",
            {"duplicate_rows": duplicate_count, **row_scope}, heuristic=True,
        ))
    checks.append(_check("dataset.duplicates", _status(duplicate_findings),
                         "Exact duplicate rows were checked.", findings=duplicate_findings,
                         metrics={"duplicate_rows": duplicate_count}, limitations=limitations))

    role_findings: list[AuditFinding] = []
    for column in p["constant_columns"]:
        role = p["roles"][column]
        role_findings.append(_finding(
            "dataset.constant_columns", str(column),
            FindingSeverity.MEDIUM if role == "empty" else FindingSeverity.LOW,
            "Empty column" if role == "empty" else "Constant column",
            "This column provides no varying observed values in the inspected data.",
            {"column": str(column), "role": role, **row_scope},
        ))
    checks.append(_check("dataset.constant_columns", _status(role_findings),
                         "Empty and constant columns were inspected.", findings=role_findings,
                         metrics={"columns": [str(c) for c in p["constant_columns"]]},
                         limitations=limitations))

    identifier_findings = [
        _finding("dataset.identifiers", str(column), FindingSeverity.MEDIUM,
                 "Identifier-like column", "A name or value pattern resembles an identifier; review whether it should be a model feature.",
                 {"column": str(column), "role": "identifier", **row_scope}, heuristic=True)
        for column in p["identifier_columns"]
    ]
    checks.append(_check("dataset.identifiers", _status(identifier_findings),
                         "Identifier-like columns were screened with name and uniqueness heuristics.",
                         findings=identifier_findings,
                         metrics={"columns": [str(c) for c in p["identifier_columns"]]},
                         limitations=limitations))

    cardinality_findings: list[AuditFinding] = []
    for column, row in p["categorical_summary"].iterrows():
        unique = int(row["unique"])
        if unique > config.MAX_ONEHOT_CATEGORIES:
            cardinality_findings.append(_finding(
                "dataset.high_cardinality", str(column), FindingSeverity.MEDIUM,
                "High-cardinality categorical column",
                "Category count exceeds the application's one-hot preprocessing limit; confirm encoding and feature suitability.",
                {"column": str(column), "unique_values": unique,
                 "application_category_limit": config.MAX_ONEHOT_CATEGORIES, **row_scope}, heuristic=True,
            ))
    checks.append(_check("dataset.high_cardinality", _status(cardinality_findings),
                         "Categorical cardinality was compared with the preprocessing limit.",
                         findings=cardinality_findings,
                         metrics={"columns_above_limit": len(cardinality_findings)}, limitations=limitations))

    checks.append(_check(
        "dataset.data_types", CheckStatus.NOT_EVALUATED,
        "Observed data types are recorded, but no expected schema was provided to validate them.",
        metrics={"dtype_counts": dict(Counter(map(str, df.dtypes)))},
        limitations=("Data type validity cannot be determined without an expected schema.",),
    ))

    if request.target_column is None:
        checks.append(_check("dataset.class_imbalance", CheckStatus.NOT_EVALUATED,
                             "Select a target column to assess class balance."))
        checks.append(_check("dataset.target_distribution", CheckStatus.NOT_EVALUATED,
                             "Select a target column to summarize its distribution."))
        checks.append(_check("dataset.target_leakage", CheckStatus.NOT_EVALUATED,
                             "Select a target column to screen for potential leakage."))
    elif target is None:
        checks.append(_check("dataset.class_imbalance", CheckStatus.NOT_EVALUATED,
                             "The selected target column is not present in the dataset."))
        checks.append(_check("dataset.target_distribution", CheckStatus.NOT_EVALUATED,
                             "The selected target column is not present in the dataset."))
        checks.append(_check("dataset.target_leakage", CheckStatus.NOT_EVALUATED,
                             "The selected target column is not present in the dataset."))
    else:
        try:
            task = infer_task(sampled[target])
        except ValueError:
            task = None
        if task != "classification":
            checks.append(_check("dataset.class_imbalance", CheckStatus.NOT_APPLICABLE,
                                 "Class imbalance applies to classification targets."))
            target_values = pd.to_numeric(sampled[target], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            ).dropna()
            if target_values.empty:
                checks.append(_check("dataset.target_distribution", CheckStatus.NOT_EVALUATED,
                                     "No finite numeric target values are available to summarize."))
            else:
                quantiles = target_values.quantile([0.25, 0.5, 0.75])
                checks.append(_check(
                    "dataset.target_distribution", CheckStatus.PASSED,
                    "Numeric target distribution was summarized with aggregate quantiles.",
                    metrics={"observed": int(len(target_values)), "q1": float(quantiles.loc[0.25]),
                             "median": float(quantiles.loc[0.5]), "q3": float(quantiles.loc[0.75])},
                    limitations=limitations,
                ))
        else:
            counts = sampled[target].value_counts(dropna=True)
            majority_share = float(counts.iloc[0] / counts.sum()) if counts.sum() else 0.0
            imbalance_findings = []
            if majority_share > 0.75:
                imbalance_findings.append(_finding(
                    "dataset.class_imbalance", "majority_class", FindingSeverity.MEDIUM,
                    "Potential class imbalance",
                    "A majority class may make accuracy less informative; review class-wise metrics and sampling strategy.",
                    {"classes_observed": int(len(counts)), "majority_share": majority_share, **row_scope},
                    heuristic=True,
                ))
            checks.append(_check("dataset.class_imbalance", _status(imbalance_findings),
                                 "Classification target distribution was inspected.",
                                 findings=imbalance_findings,
                                 metrics={"classes_observed": int(len(counts)),
                                          "majority_share": majority_share}, limitations=limitations))
            class_shares = sorted((counts / counts.sum()).astype(float).tolist(), reverse=True) if counts.sum() else []
            checks.append(_check(
                "dataset.target_distribution", CheckStatus.PASSED,
                "Classification target distribution was summarized without exporting class labels.",
                metrics={"classes_observed": int(len(counts)), "class_counts": sorted(
                    (int(value) for value in counts.tolist()), reverse=True),
                    "class_shares": class_shares},
                limitations=limitations,
            ))

        leakage_messages = p["leakage_risks"]
        leakage_findings = [
            _finding("dataset.target_leakage", f"heuristic_{index}", FindingSeverity.HIGH,
                     "Potential target leakage", message,
                     {"observation": message, **row_scope}, heuristic=True)
            for index, message in enumerate(leakage_messages, start=1)
        ]
        checks.append(_check("dataset.target_leakage", _status(leakage_findings),
                             "Target-related name, correlation, and category-purity heuristics were checked.",
                             findings=leakage_findings,
                             metrics={"potential_signals": len(leakage_findings)},
                             limitations=("Signals are heuristics and do not prove leakage.", *limitations)))

    # Outliers are reported as an IQR screening signal, never as invalid rows.
    outlier_findings: list[AuditFinding] = []
    for name in list(sampled.select_dtypes(include="number").columns)[:MAX_PROFILE_COLUMNS]:
        values = sampled[name].replace([np.inf, -np.inf], np.nan).dropna()
        if len(values) < 4:
            continue
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        if not np.isfinite(iqr) or iqr <= 0:
            continue
        count = int(((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)).sum())
        rate = count / len(values)
        if rate >= 0.05:
            outlier_findings.append(_finding(
                "dataset.outliers", str(name), FindingSeverity.LOW,
                "Potential outliers by IQR rule",
                "The IQR rule flags unusual values for review; flagged values are not necessarily errors.",
                {"column": str(name), "flagged_count": count, "observed_rate": rate, **row_scope},
                heuristic=True,
            ))
    checks.append(_check("dataset.outliers", _status(outlier_findings),
                         "Numeric outlier screening used the 1.5×IQR rule.",
                         findings=outlier_findings, metrics={"columns_flagged": len(outlier_findings)},
                         limitations=("Outlier flags are descriptive and do not establish invalid data.", *limitations)))

    checks.append(_check("dataset.distributions", CheckStatus.PASSED,
                         "Aggregate feature distributions and numeric correlations were summarized.",
                         metrics=_safe_column_summary(sampled, p), limitations=limitations))
    return tuple(checks)
