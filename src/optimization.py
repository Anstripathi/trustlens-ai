"""Bounded randomized hyperparameter search on the fit split only."""
from __future__ import annotations

from dataclasses import replace
from time import perf_counter

import numpy as np
from sklearn.base import clone
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, ParameterGrid, RandomizedSearchCV, StratifiedKFold, KFold

from . import config
from .data_validation import ValidationError
from .evaluate import evaluate

MAX_SEARCH_BUDGET = 20
MAX_CV_FOLDS = 5


def _search_space(result):
    name = result.config.model_name
    if name == "Logistic Regression":
        return {"model__C": np.geomspace(0.001, 1000.0, 101).tolist()}
    if name in ("Decision Tree", "Random Forest"):
        space = {
            "model__max_depth": list(range(*config.HP_BOUNDS["max_depth"])) + [config.HP_BOUNDS["max_depth"][1]],
            "model__min_samples_leaf": [1, 2, 5, 10, 20, 35, 50],
        }
        if name == "Random Forest":
            space["model__n_estimators"] = [50, 100, 150]
        return space
    raise ValidationError(f"Optimization is not configured for {name}.")


def _complexity(estimator):
    if hasattr(estimator, "estimators_"):
        return int(sum(tree.tree_.node_count for tree in estimator.estimators_)), "tree nodes"
    if hasattr(estimator, "tree_"):
        return int(estimator.tree_.node_count), "tree nodes"
    if hasattr(estimator, "coef_"):
        return int((estimator.coef_ != 0).sum()), "non-zero coefficients"
    return None, "unavailable"


def optimize_model(result, search_budget: int = 10, cv_folds: int = 3, search_method: str = "randomized") -> dict:
    """Search model parameters using only fit rows, then score on calibration rows.

    The returned optimized TrainResult retains the original untouched test split.
    Applying it to the session is an explicit UI action.
    """
    if isinstance(search_budget, bool) or not isinstance(search_budget, int) or not 1 <= search_budget <= MAX_SEARCH_BUDGET:
        raise ValidationError(f"Search budget must be between 1 and {MAX_SEARCH_BUDGET} iterations.")
    if isinstance(cv_folds, bool) or not isinstance(cv_folds, int) or not 2 <= cv_folds <= MAX_CV_FOLDS:
        raise ValidationError(f"Cross-validation folds must be between 2 and {MAX_CV_FOLDS}.")
    if search_method not in ("randomized", "grid"):
        raise ValidationError("Search method must be randomized or grid.")
    if result.task == "classification":
        class_counts = result.y_fit.value_counts()
        if len(class_counts) < 2 or class_counts.min() < cv_folds:
            raise ValidationError("Each class needs at least one fit row per cross-validation fold.")
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=result.config.seed)
        scoring = (make_scorer(f1_score, pos_label=result.classes[1], zero_division=0)
                   if len(result.classes) == 2 else "f1_macro")
    elif result.task == "regression":
        if len(result.X_fit) < cv_folds:
            raise ValidationError("There are too few fit rows for the selected cross-validation folds.")
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=result.config.seed)
        scoring = "neg_root_mean_squared_error"
    else:
        raise ValidationError(f"Unsupported task: {result.task!r}.")

    started = perf_counter()
    if search_method == "grid":
        # Bound the cartesian grid deterministically; no unbounded grid can be requested.
        all_candidates = list(ParameterGrid(_search_space(result)))
        indices = np.linspace(0, len(all_candidates) - 1,
                              num=min(search_budget, len(all_candidates)), dtype=int)
        candidates = [{name: [value] for name, value in all_candidates[index].items()} for index in indices]
        search = GridSearchCV(
            clone(result.pipeline), candidates, scoring=scoring, cv=cv, refit=True,
            n_jobs=1, error_score="raise", return_train_score=False,
        )
    else:
        search = RandomizedSearchCV(
            clone(result.pipeline), _search_space(result), n_iter=search_budget,
            scoring=scoring, cv=cv, refit=True, n_jobs=1,
            random_state=result.config.seed, error_score="raise", return_train_score=False,
        )
    try:
        # The preprocessing pipeline is cloned and fit inside each CV fold.
        search.fit(result.X_fit, result.y_fit)
    except Exception:  # noqa: BLE001 - do not echo row or feature values into UI errors
        raise ValidationError("Hyperparameter search failed; try fewer folds or a smaller search budget.") from None
    fit_seconds = perf_counter() - started

    best_pipeline = search.best_estimator_
    predict_start = perf_counter()
    best_pipeline.predict(result.X_cal)
    predict_ms_per_row = (perf_counter() - predict_start) * 1000 / max(1, len(result.X_cal))
    validation_metrics = evaluate(best_pipeline, result.X_cal, result.y_cal, result.task)
    best_params = {name.removeprefix("model__"): value for name, value in search.best_params_.items()}
    optimized_config = replace(result.config, hyperparams=best_params)
    optimized_result = replace(result, pipeline=best_pipeline, config=optimized_config)
    complexity, complexity_unit = _complexity(best_pipeline.named_steps["model"])
    return {
        "result": optimized_result,
        "baseline_validation": evaluate(result.pipeline, result.X_cal, result.y_cal, result.task),
        "optimized_validation": validation_metrics,
        "best_params": best_params,
        "cv_score": float(search.best_score_),
        "cv_scoring": scoring,
        "search_budget": search_budget,
        "search_method": search_method,
        "cv_folds": cv_folds,
        "fit_seconds": fit_seconds,
        "predict_ms_per_row": predict_ms_per_row,
        "complexity": complexity,
        "complexity_unit": complexity_unit,
        "note": "Search and cross-validation used fit rows only; validation metrics use the calibration split. The test split was not used for tuning or selection.",
    }
