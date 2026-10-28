import numpy as np
import pandas as pd
import pytest
from helpers import clf_df, reg_df
from src.explain import global_importance, local_explanation
from src.train import TrainConfig, train_model


def test_global_importance_finds_informative_feature():
    r = train_model(reg_df(), TrainConfig("y", "regression", "Random Forest"))
    g = global_importance(r)
    assert set(g["table"].feature) == set(r.features)          # names align with ORIGINAL columns
    assert g["table"].feature.iloc[0] == "a"


def test_global_importance_classification():
    r = train_model(clf_df(), TrainConfig("y", "classification", "Random Forest"))
    assert global_importance(r)["table"].feature.iloc[0] == "a"


def test_local_explanation_matches_prediction_and_linear_math():
    r = train_model(reg_df(), TrainConfig("y", "regression", "Linear Regression"))
    row = r.X_test.iloc[[0]]
    ex = local_explanation(r, row)
    assert np.isclose(ex["prediction"], r.pipeline.predict(row)[0]) and np.isclose(ex["model_output"], ex["prediction"])
    c = ex["contributions"].set_index("feature")
    ref_a = r.X_fit["a"].median()
    slope = r.pipeline.predict(pd.DataFrame({"a": [1.0], "b": [row.b.iloc[0]]}))[0] - \
        r.pipeline.predict(pd.DataFrame({"a": [0.0], "b": [row.b.iloc[0]]}))[0]
    assert np.isclose(c.loc["a", "contribution"], slope * (row.a.iloc[0] - ref_a))


def test_local_explanation_classification_output_is_probability_of_positive_class():
    r = train_model(clf_df(), TrainConfig("y", "classification", "Logistic Regression"))
    row = r.X_test.iloc[[3]]
    ex = local_explanation(r, row)
    assert np.isclose(ex["model_output"], r.pipeline.predict_proba(row)[0, 1])
    assert "yes" in ex["output_explained"]


def test_local_requires_single_row():
    r = train_model(reg_df(), TrainConfig("y", "regression", "Decision Tree"))
    with pytest.raises(ValueError):
        local_explanation(r, r.X_test.iloc[:2])
