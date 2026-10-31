import pytest
from dataclasses import replace

from helpers import clf_df, reg_df
from src.counterfactual import CounterfactualError, generate_counterfactuals
from src.train import TrainConfig, train_model


def test_classification_counterfactual_finds_desired_class_without_changing_immutable_feature():
    result = train_model(clf_df(nan=False), TrainConfig("y", "classification", "Logistic Regression"))
    row = result.X_test.iloc[[0]]
    current = result.pipeline.predict(row)[0]
    desired = next(value for value in result.classes if value != current)
    out = generate_counterfactuals(result, row, desired_class=desired, immutable_features={"b"}, max_candidates=100)
    assert out["found"]
    assert (out["candidates"]["prediction"] == desired).all()
    assert (out["candidates"]["b"] == row.iloc[0]["b"]).all()
    assert out["evaluated"] <= 100


def test_regression_counterfactual_obeys_target_and_feature_ranges():
    result = train_model(reg_df(), TrainConfig("y", "regression", "Linear Regression"))
    row = result.X_test.iloc[[0]]
    fit_min, fit_max = result.X_fit["a"].min(), result.X_fit["a"].max()
    target_row = row.copy()
    target_row["a"] = float(result.X_fit["a"].min())
    target = float(result.pipeline.predict(target_row)[0])
    out = generate_counterfactuals(
        result, row, target_range=(target - 0.1, target + 0.1), immutable_features={"b"},
        feature_ranges={"a": (float(fit_min), float(fit_max))},
        max_candidates=150, max_changes=2,
    )
    assert out["found"]
    assert out["candidates"]["prediction"].between(target - 2.0, target + 2.0).all()
    assert out["candidates"]["a"].between(fit_min, fit_max).all()


def test_counterfactual_reports_no_candidate_when_every_feature_is_immutable():
    result = train_model(clf_df(), TrainConfig("y", "classification", "Decision Tree"))
    row = result.X_test.iloc[[0]]
    desired = next(value for value in result.classes if value != result.pipeline.predict(row)[0])
    out = generate_counterfactuals(result, row, desired_class=desired, immutable_features=set(result.features))
    assert not out["found"]
    assert out["evaluated"] == 0


@pytest.mark.parametrize("kwargs", [
    {"max_candidates": 201}, {"max_changes": 4}, {"target_range": (1.0, 0.0)},
    {"immutable_features": {"unknown"}},
])
def test_counterfactual_validates_user_constraints(kwargs):
    result = train_model(reg_df(), TrainConfig("y", "regression", "Linear Regression"))
    with pytest.raises(CounterfactualError):
        params = {"target_range": (0, 1), **kwargs}
        generate_counterfactuals(result, result.X_test.iloc[[0]], **params)


def test_counterfactual_rejects_categorical_numeric_ranges():
    result = train_model(clf_df(), TrainConfig("y", "classification", "Logistic Regression"))
    row = result.X_test.iloc[[0]]
    desired = next(value for value in result.classes if value != result.pipeline.predict(row)[0])
    with pytest.raises(CounterfactualError, match="numeric features"):
        generate_counterfactuals(result, row, desired_class=desired, feature_ranges={"b": (0, 1)})


def test_counterfactual_reports_unsupported_task():
    result = train_model(reg_df(), TrainConfig("y", "regression", "Linear Regression"))
    with pytest.raises(CounterfactualError, match="Unsupported prediction task"):
        generate_counterfactuals(replace(result, task="unsupported"), result.X_test.iloc[[0]], target_range=(0, 1))
