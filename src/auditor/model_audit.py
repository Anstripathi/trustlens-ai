"""Evaluation-only audit adapters for the currently selected trained model."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

from ..evaluate import evaluate
from ..train import TrainConfig, compare_models
from .schemas import AuditCategory, AuditCheck, AuditFinding, AuditRequest, CheckStatus, FindingSeverity

MAX_CV_ROWS = 5_000
CV_FOLDS = 3
MAX_MODEL_METRIC_ROWS = 10_000
MAX_BENCHMARK_ROWS = 10_000


def _check(check_id, status, summary, *, findings=(), metrics=None, limitations=()):
    return AuditCheck(check_id, AuditCategory.MODEL, status, summary, tuple(findings),
                      dict(metrics or {}), tuple(limitations))


def _finding(check_id, severity, title, explanation, evidence, suffix):
    return AuditFinding(f"{check_id}.{suffix}", AuditCategory.MODEL, severity, title,
                        explanation, evidence, source_check=check_id, heuristic=True)


def _not_evaluated(reason):
    return (
        _check("model.performance", CheckStatus.NOT_EVALUATED, reason),
        _check("model.generalization", CheckStatus.NOT_EVALUATED, reason),
        _check("model.split_integrity", CheckStatus.NOT_EVALUATED, reason),
        _check("model.cross_validation", CheckStatus.NOT_EVALUATED, reason),
        _check("model.validation_comparison", CheckStatus.NOT_EVALUATED,
               "Optional model comparison was not requested."),
    )


def _scalar_metrics(result, X, y):
    raw = evaluate(result.pipeline, X, y, result.task)
    allowed = ("accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc",
               "average_precision", "brier", "mae", "mse", "rmse", "r2",
               "majority_baseline_accuracy", "mean_baseline_rmse")
    return {name: float(raw[name]) if raw.get(name) is not None and np.isfinite(raw[name]) else None
            for name in allowed if name in raw}


def _classwise_metrics(result, X_test, y_test):
    predictions = result.pipeline.predict(X_test)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, predictions, labels=result.classes, zero_division=0,
    )
    # Labels are represented by ordinals so report output never copies user class values.
    return {
        f"class_{index + 1}": {"precision": float(precision[index]), "recall": float(recall[index]),
                               "f1": float(f1[index]), "support": int(support[index])}
        for index in range(len(result.classes))
    }


def audit_model(request: AuditRequest) -> tuple[AuditCheck, ...]:
    """Inspect a current result without changing it or tuning against test data."""
    result = request.model
    if result is None:
        return _not_evaluated("No trained model is available.")
    if request.dataset is None or request.dataset_fingerprint is None:
        return _not_evaluated("A matching active dataset is required.")
    if request.model_fingerprint != request.dataset_fingerprint:
        return _not_evaluated("The active model was trained on a different dataset.")
    if not hasattr(result, "pipeline") or not hasattr(result, "X_test"):
        return (_check("model.performance", CheckStatus.FAILED,
                       "The active model result has an unsupported structure."),
                _check("model.generalization", CheckStatus.NOT_EVALUATED,
                       "Generalization could not be checked without a supported model result."),
                _check("model.split_integrity", CheckStatus.NOT_EVALUATED,
                       "Split integrity could not be checked without a supported model result."),
                _check("model.cross_validation", CheckStatus.NOT_EVALUATED,
                       "Cross-validation could not be checked without a supported model result."),
                _check("model.validation_comparison", CheckStatus.NOT_EVALUATED,
                       "No validation comparison was produced."))

    checks = []
    try:
        splits = {"fit": result.X_fit, "validation": result.X_cal, "test": result.X_test}
        overlap_counts = {}
        split_names = list(splits)
        for left_index, left_name in enumerate(split_names):
            for right_name in split_names[left_index + 1:]:
                overlap_counts[f"{left_name}_{right_name}"] = int(
                    len(splits[left_name].index.intersection(splits[right_name].index))
                )
        overlapping = {name: count for name, count in overlap_counts.items() if count}
        findings = []
        if overlapping:
            findings.append(_finding(
                "model.split_integrity", FindingSeverity.HIGH, "Split row indices overlap",
                "Stored fit, validation, and test split indices overlap; confirm that evaluation rows were not used during fitting or selection.",
                {"overlap_counts": overlapping}, "index_overlap",
            ))
        else:
            # A bounded sample screens exact duplicate feature rows across fit and test.
            fit = result.X_fit
            test = result.X_test
            if len(fit) > 10_000:
                fit = fit.sample(10_000, random_state=result.config.seed)
            if len(test) > 10_000:
                test = test.sample(10_000, random_state=result.config.seed)
            fit_hashes = set(pd.util.hash_pandas_object(fit, index=False).tolist())
            test_hashes = set(pd.util.hash_pandas_object(test, index=False).tolist())
            duplicate_cross_split = len(fit_hashes & test_hashes)
            if duplicate_cross_split:
                findings.append(_finding(
                    "model.split_integrity", FindingSeverity.LOW, "Repeated feature rows cross fit and test splits",
                    "Exact feature duplicates were found across a bounded split sample; repeated records may be valid, but can make evaluation optimistic.",
                    {"matching_feature_hashes": int(duplicate_cross_split),
                     "fit_rows_sampled": int(len(fit)), "test_rows_sampled": int(len(test))},
                    "duplicate_features",
                ))
        checks.append(_check(
            "model.split_integrity", CheckStatus.FINDING if findings else CheckStatus.PASSED,
            "Stored split indices and a bounded feature-duplicate screen were checked.",
            findings=findings, metrics={"index_overlap_counts": overlap_counts,
                                        "fit_rows": int(len(result.X_fit)),
                                        "validation_rows": int(len(result.X_cal)),
                                        "test_rows": int(len(result.X_test))},
            limitations=("No detected overlap is not proof that the split strategy is appropriate.",
                         "Repeated features are a heuristic signal and may be valid observations."),
        ))
    except Exception:  # noqa: BLE001 - no row contents in failure messages
        checks.append(_check("model.split_integrity", CheckStatus.FAILED,
                             "Stored split integrity could not be inspected."))
    try:
        X_test, y_test = result.X_test, result.y_test
        X_fit, y_fit = result.X_fit, result.y_fit
        if len(X_test) > MAX_MODEL_METRIC_ROWS:
            X_test = X_test.sample(MAX_MODEL_METRIC_ROWS, random_state=result.config.seed)
            y_test = y_test.loc[X_test.index]
        if len(X_fit) > MAX_MODEL_METRIC_ROWS:
            X_fit = X_fit.sample(MAX_MODEL_METRIC_ROWS, random_state=result.config.seed)
            y_fit = y_fit.loc[X_fit.index]
        test_metrics = _scalar_metrics(result, X_test, y_test)
        classwise = _classwise_metrics(result, X_test, y_test) if result.task == "classification" else {}
        fit_metrics = _scalar_metrics(result, X_fit, y_fit)
        if result.task == "classification":
            from sklearn.metrics import confusion_matrix
            predictions = result.pipeline.predict(X_test)
            cm = confusion_matrix(y_test, predictions, labels=result.classes)
            classwise_confusion = {"labels": [f"class_{i + 1}" for i in range(len(result.classes))],
                                   "matrix": cm.astype(int).tolist()}
        else:
            classwise_confusion = None
        findings = []
        primary = "f1" if result.task == "classification" else "rmse"
        test_primary = test_metrics.get(primary)
        fit_primary = fit_metrics.get(primary)
        if result.task == "classification" and test_primary is not None and test_primary < 0.60:
            findings.append(_finding(
                "model.performance", FindingSeverity.MEDIUM, "Low held-out F1 score",
                "The selected model's held-out F1 is below the audit's 0.60 review threshold; check class-wise results and task requirements.",
                {"metric": primary, "value": test_primary, "review_threshold": 0.60,
                 "test_rows": int(len(y_test))}, "low_f1",
            ))
        if result.task == "regression" and test_primary is not None and fit_primary is not None:
            baseline = test_metrics.get("mean_baseline_rmse")
            if baseline and test_primary > baseline:
                findings.append(_finding(
                    "model.performance", FindingSeverity.MEDIUM, "Model trails the mean baseline",
                    "Held-out RMSE exceeds the baseline that always predicts the training mean; review feature and target suitability.",
                    {"test_rmse": test_primary, "mean_baseline_rmse": baseline,
                     "test_rows": int(len(y_test))}, "baseline_gap",
                ))
        performance_status = CheckStatus.FINDING if findings else CheckStatus.PASSED
        sample_limitations = []
        if len(y_test) < 30:
            sample_limitations.append("The held-out test split contains fewer than 30 rows; estimates may be noisy.")
        if len(y_test) < len(result.y_test):
            sample_limitations.append(f"Metrics use a deterministic {len(y_test):,}-row sample of the held-out test split.")
        if len(y_fit) < len(result.y_fit):
            sample_limitations.append(f"Fit diagnostics use a deterministic {len(y_fit):,}-row sample of fit rows.")
        checks.append(_check(
            "model.performance", performance_status,
            "Metrics were evaluated on the model's held-out test split.", findings=findings,
            metrics={"task": result.task, "test_rows": int(len(y_test)),
                     "test_split_rows": int(len(result.y_test)),
                     "test_metrics": test_metrics, "classwise_test_metrics": classwise,
                     "confusion_matrix": classwise_confusion,
                     "primary_metric": primary},
            limitations=("Threshold findings are review signals, not universal quality standards.", *sample_limitations),
        ))

        gap_findings = []
        if result.task == "classification":
            train_value, test_value = fit_metrics.get("f1"), test_metrics.get("f1")
            gap = train_value - test_value if train_value is not None and test_value is not None else None
            gap_name = "f1_train_minus_test"
        else:
            train_value, test_value = fit_metrics.get("rmse"), test_metrics.get("rmse")
            gap = test_value / max(train_value, 1e-12) if train_value is not None and test_value is not None else None
            gap_name = "test_to_fit_rmse_ratio"
        if gap is not None and ((result.task == "classification" and gap >= 0.15)
                                or (result.task == "regression" and gap >= 1.5)):
            gap_findings.append(_finding(
                "model.generalization", FindingSeverity.MEDIUM, "Potential generalization gap",
                "Training and held-out metrics differ beyond a heuristic review threshold; inspect data splits and deployment similarity.",
                {"gap_metric": gap_name, "gap_value": float(gap), "fit_rows": int(len(y_fit)),
                 "test_rows": int(len(y_test))}, "performance_gap",
            ))
        checks.append(_check(
            "model.generalization", CheckStatus.FINDING if gap_findings else CheckStatus.PASSED,
            "Fit-split and held-out metrics were compared; fit metrics are in-sample.",
            findings=gap_findings,
            metrics={"fit_metrics": fit_metrics, "test_metrics": test_metrics,
                     "gap_metric": gap_name, "gap_value": gap},
            limitations=("The fit score is an in-sample diagnostic, not an estimate of generalization.",
                         "A small gap does not establish generalization to future data."),
        ))
    except Exception:  # noqa: BLE001 - keep model/input details out of the report
        checks.extend((
            _check("model.performance", CheckStatus.FAILED,
                   "Held-out model metrics could not be calculated."),
            _check("model.generalization", CheckStatus.FAILED,
                   "Generalization metrics could not be calculated."),
        ))

    if not request.run_cross_validation:
        checks.append(_check("model.cross_validation", CheckStatus.NOT_EVALUATED,
                             "Cross-validation refits model clones and was not explicitly requested."))
    else:
        try:
            X_cv, y_cv = result.X_fit, result.y_fit
            if len(X_cv) > MAX_CV_ROWS:
                X_cv = X_cv.sample(MAX_CV_ROWS, random_state=result.config.seed)
                y_cv = y_cv.loc[X_cv.index]
            if result.task == "classification":
                counts = y_cv.value_counts()
                if len(counts) < 2 or int(counts.min()) < CV_FOLDS:
                    raise ValueError("insufficient class support")
                splitter = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                                           random_state=result.config.seed)
                scoring = "f1_macro"
            elif result.task == "regression" and len(y_cv) >= CV_FOLDS:
                splitter = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=result.config.seed)
                scoring = "neg_root_mean_squared_error"
            else:
                raise ValueError("insufficient rows")
            scores = cross_val_score(clone(result.pipeline), X_cv, y_cv, scoring=scoring,
                                     cv=splitter, n_jobs=1, error_score="raise")
            values = -scores if scoring == "neg_root_mean_squared_error" else scores
            checks.append(_check(
                "model.cross_validation", CheckStatus.PASSED,
                "Three-fold cross-validation refit pipeline clones on fit rows only.",
                metrics={"scoring": "rmse" if scoring.startswith("neg_") else scoring,
                         "folds": CV_FOLDS, "rows": int(len(X_cv)),
                         "mean": float(np.mean(values)), "std": float(np.std(values, ddof=1))},
                limitations=("Cross-validation describes this fit split and is not test-set evaluation.",),
            ))
        except Exception:  # noqa: BLE001
            checks.append(_check("model.cross_validation", CheckStatus.NOT_EVALUATED,
                                 "Cross-validation was unavailable for the current fit data or model."))

    if request.benchmark_models:
        try:
            config = result.config
            benchmark_data = request.dataset
            benchmark_sampled = len(benchmark_data) > MAX_BENCHMARK_ROWS
            if benchmark_sampled:
                benchmark_data = benchmark_data.sample(MAX_BENCHMARK_ROWS, random_state=config.seed)
            comparison_config = TrainConfig(
                target=config.target, task=config.task, model_name=config.model_name,
                features=list(result.features), test_size=config.test_size, cal_size=config.cal_size,
                seed=config.seed, hyperparams={},
            )
            comparison = compare_models(benchmark_data, comparison_config)
            metrics_by_model = {}
            metric_names = (("accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc")
                            if result.task == "classification" else ("mae", "mse", "rmse", "r2"))
            for name, row in comparison.iterrows():
                metrics_by_model[str(name)] = {
                    metric: (float(row[metric]) if row.get(metric) is not None and
                             np.isfinite(row[metric]) else None)
                    for metric in metric_names if metric in row
                }
            checks.append(_check(
                "model.validation_comparison", CheckStatus.PASSED,
                "Supported candidate models were compared on validation rows; the test split was not used for selection.",
                metrics={"candidate_count": len(metrics_by_model), "metrics": metrics_by_model,
                         "rows_used": int(len(benchmark_data)),
                         "full_dataset_rows": int(len(request.dataset)),
                         "sampled": benchmark_sampled,
                         "selection_split": "calibration/validation", "active_model_changed": False},
                limitations=("This explicit opt-in fits candidate models for comparison; it does not replace the active model.",),
            ))
        except Exception:  # noqa: BLE001
            checks.append(_check("model.validation_comparison", CheckStatus.FAILED,
                                 "Validation-split model comparison could not be completed."))
    else:
        checks.append(_check("model.validation_comparison", CheckStatus.NOT_EVALUATED,
                             "Candidate models were not retrained; comparison requires explicit opt-in."))
    return tuple(checks)
