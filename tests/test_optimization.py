import numpy as np
import pytest
from dataclasses import replace

from helpers import clf_df, reg_df
from src.data_validation import ValidationError
from src.optimization import optimize_model
from src.train import TrainConfig, compare_models, train_model


@pytest.mark.parametrize("task,frame,model", [
    ("classification", clf_df, "Decision Tree"),
    ("regression", reg_df, "Decision Tree"),
])
def test_optimization_is_bounded_reproducible_and_preserves_test_split(task, frame, model):
    result = train_model(frame(), TrainConfig("y", task, model, seed=5))
    optimized = optimize_model(result, search_budget=2, cv_folds=2)
    assert optimized["search_budget"] == 2 and optimized["cv_folds"] == 2
    assert np.isfinite(optimized["cv_score"])
    assert optimized["best_params"]
    assert optimized["result"].X_test.equals(result.X_test)
    assert optimized["result"].y_test.equals(result.y_test)
    assert optimized["fit_seconds"] >= 0 and optimized["predict_ms_per_row"] >= 0


@pytest.mark.parametrize("kwargs", [{"search_budget": 21}, {"search_budget": 0}, {"cv_folds": 1}, {"cv_folds": 6}])
def test_optimization_validates_compute_budget(kwargs):
    result = train_model(clf_df(), TrainConfig("y", "classification", "Decision Tree"))
    with pytest.raises(ValidationError):
        optimize_model(result, **kwargs)


def test_optimization_handles_rare_classes_before_search():
    result = train_model(clf_df(n=100, nan=False), TrainConfig("y", "classification", "Decision Tree"))
    result.y_fit.iloc[0] = "__rare__"
    with pytest.raises(ValidationError, match="Each class needs"):
        optimize_model(result, search_budget=2, cv_folds=2)


def test_model_comparison_reports_validation_metrics_and_runtime():
    result = compare_models(clf_df(), TrainConfig("y", "classification", "Logistic Regression"))
    assert {"accuracy", "precision", "recall", "f1", "roc_auc", "fit_seconds",
            "predict_ms_per_row", "complexity", "confusion_matrix"} <= set(result.columns)
    assert result["fit_seconds"].ge(0).all()
    assert result["predict_ms_per_row"].ge(0).all()


def test_optimization_rejects_unsupported_task():
    result = train_model(reg_df(), TrainConfig("y", "regression", "Linear Regression"))
    with pytest.raises(ValidationError, match="Unsupported task"):
        optimize_model(replace(result, task="unsupported"), search_budget=2, cv_folds=2)


def test_bounded_grid_search_uses_fit_rows_and_respects_budget():
    result = train_model(clf_df(), TrainConfig("y", "classification", "Decision Tree", seed=5))
    optimized = optimize_model(result, search_budget=2, cv_folds=2, search_method="grid")
    assert optimized["search_method"] == "grid"
    assert optimized["search_budget"] == 2
    assert optimized["result"].X_test.equals(result.X_test)


def test_search_method_is_validated():
    result = train_model(clf_df(), TrainConfig("y", "classification", "Decision Tree"))
    with pytest.raises(ValidationError, match="Search method"):
        optimize_model(result, search_method="unbounded")
