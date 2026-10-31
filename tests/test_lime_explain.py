import numpy as np
import pandas as pd
import pytest
from dataclasses import replace

from helpers import clf_df, reg_df
from src.lime_explain import LimeUnavailableError, explain_lime
from src.train import TrainConfig, train_model


@pytest.mark.parametrize("model", ["Logistic Regression", "Decision Tree", "Random Forest"])
def test_lime_explains_classification_with_original_feature_names(model):
    result = train_model(clf_df(), TrainConfig("y", "classification", model))
    out = explain_lime(result, result.X_test.iloc[[0]], num_samples=200)
    assert out["output_name"].startswith("probability of class")
    assert np.isfinite(out["model_output"])
    assert set(out["contributions"].feature) <= set(result.features)
    assert 0 <= out["local_r2"] <= 1


@pytest.mark.parametrize("model", ["Linear Regression", "Decision Tree", "Random Forest"])
def test_lime_explains_regression(model):
    result = train_model(reg_df(), TrainConfig("y", "regression", model))
    out = explain_lime(result, result.X_test.iloc[[0]], num_samples=200)
    assert out["output_name"] == "predicted value"
    assert np.isfinite(out["local_prediction"])
    assert set(out["contributions"].feature) <= set(result.features)


def test_lime_rejects_invalid_rows_and_budget():
    result = train_model(reg_df(), TrainConfig("y", "regression", "Linear Regression"))
    with pytest.raises(LimeUnavailableError, match="one observation"):
        explain_lime(result, result.X_test.iloc[:2])
    with pytest.raises(LimeUnavailableError, match="sample count"):
        explain_lime(result, result.X_test.iloc[[0]], num_samples=50)
    with pytest.raises(LimeUnavailableError, match="missing feature"):
        explain_lime(result, pd.DataFrame({"a": [1.0]}))


def test_lime_is_reproducible_with_fixed_seed():
    result = train_model(clf_df(), TrainConfig("y", "classification", "Logistic Regression", seed=7))
    row = result.X_test.iloc[[0]]
    first = explain_lime(result, row, num_samples=300)
    second = explain_lime(result, row, num_samples=300)
    assert np.allclose(first["contributions"].weight, second["contributions"].weight)


def test_lime_reports_unsupported_task_and_empty_features():
    result = train_model(reg_df(), TrainConfig("y", "regression", "Linear Regression"))
    with pytest.raises(LimeUnavailableError, match="Unsupported prediction task"):
        explain_lime(replace(result, task="unsupported"), result.X_test.iloc[[0]])
    with pytest.raises(LimeUnavailableError, match="No model features"):
        explain_lime(replace(result, features=[]), result.X_test.iloc[[0]])
