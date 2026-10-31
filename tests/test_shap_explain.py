import numpy as np
import pandas as pd
import pytest
from dataclasses import replace

from helpers import clf_df, reg_df
from src.shap_explain import ShapUnavailableError, compare_shap_importance, explain_shap
from src.train import TrainConfig, train_model


@pytest.mark.parametrize("model", ["Logistic Regression", "Decision Tree", "Random Forest"])
def test_shap_explains_classification_pipeline_and_original_columns(model):
    result = train_model(clf_df(), TrainConfig("y", "classification", model))
    out = explain_shap(result, result.X_test.iloc[:2], class_label=result.classes[-1])
    assert out["values"].shape == (2, len(result.features))
    assert list(out["contributions"].columns) == result.features
    assert out["output_name"].startswith("probability of class")
    assert np.allclose(out["base_values"] + out["values"].sum(axis=1), out["predictions"], atol=0.12)
    assert set(out["importance"].feature) == set(result.features)


@pytest.mark.parametrize("model", ["Linear Regression", "Decision Tree", "Random Forest"])
def test_shap_explains_regression_pipeline(model):
    result = train_model(reg_df(), TrainConfig("y", "regression", model))
    out = explain_shap(result, result.X_test.iloc[:1])
    assert out["values"].shape == (1, len(result.features))
    assert np.allclose(out["base_values"] + out["values"].sum(axis=1), out["predictions"], atol=0.2)


def test_shap_bounds_inputs_and_rejects_missing_features():
    result = train_model(reg_df(), TrainConfig("y", "regression", "Linear Regression"))
    with pytest.raises(ShapUnavailableError, match="Background size"):
        explain_shap(result, background_size=1000)
    with pytest.raises(ShapUnavailableError, match="missing feature"):
        explain_shap(result, pd.DataFrame({"a": [1.0]}))
    with pytest.raises(ShapUnavailableError, match="no rows"):
        explain_shap(result, result.X_test.iloc[:0])


def test_shap_importance_compares_all_compatible_classifiers_on_same_rows():
    result = train_model(clf_df(), TrainConfig("y", "classification", "Decision Tree"))
    compared = compare_shap_importance(result, result.X_test.iloc[:1], class_label=result.classes[-1])
    assert {"Logistic Regression", "Decision Tree", "Random Forest"} <= set(compared["models"])
    assert set(compared["importance"].index) == set(result.features)
    assert (compared["importance"].dropna() >= 0).all().all()


def test_shap_reports_unsupported_task_and_empty_features():
    result = train_model(reg_df(), TrainConfig("y", "regression", "Linear Regression"))
    with pytest.raises(ShapUnavailableError, match="Unsupported prediction task"):
        explain_shap(replace(result, task="unsupported"), result.X_test.iloc[:1])
    with pytest.raises(ShapUnavailableError, match="No model features"):
        explain_shap(replace(result, features=[]), result.X_test.iloc[:1])
