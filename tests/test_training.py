import numpy as np
import pandas as pd
import pytest
from helpers import clf_df, reg_df
from src.data_validation import ValidationError
from src.evaluate import evaluate
from src.preprocessing import build_preprocessor, split_feature_kinds
from src.train import MODELS_BY_TASK, TrainConfig, compare_models, train_model


@pytest.mark.parametrize("model", MODELS_BY_TASK["classification"])
def test_classification_all_models(model):
    hp = {} if model == "Logistic Regression" else {"max_depth": 4}      # unpruned trees overfit noisy labels
    r = train_model(clf_df(), TrainConfig("y", "classification", model, hyperparams=hp))
    m = evaluate(r.pipeline, r.X_test, r.y_test, "classification")
    assert m["accuracy"] > 0.6 and 0.5 < m["roc_auc"] <= 1.0
    assert m["confusion_matrix"].values.sum() == len(r.y_test)


@pytest.mark.parametrize("model", MODELS_BY_TASK["regression"])
def test_regression_all_models(model):
    r = train_model(reg_df(), TrainConfig("y", "regression", model))
    m = evaluate(r.pipeline, r.X_test, r.y_test, "regression")
    assert m["r2"] > 0.5 and m["rmse"] < m["mean_baseline_rmse"]
    assert np.isclose(m["rmse"] ** 2, m["mse"])


def test_splits_disjoint_cover_and_reproducible():
    df = clf_df()
    a = train_model(df, TrainConfig("y", "classification", "Decision Tree", seed=1))
    b = train_model(df, TrainConfig("y", "classification", "Decision Tree", seed=1))
    c = train_model(df, TrainConfig("y", "classification", "Decision Tree", seed=2))
    fit, cal, test = set(a.X_fit.index), set(a.X_cal.index), set(a.X_test.index)
    assert not (fit & cal or fit & test or cal & test) and len(fit | cal | test) == len(df)
    assert list(a.X_test.index) == list(b.X_test.index) != list(c.X_test.index)
    assert (a.pipeline.predict(a.X_test) == b.pipeline.predict(b.X_test)).all()
    assert a.split_info["stratified"]
    assert abs(a.y_test.eq("yes").mean() - df.y.eq("yes").mean()) < 0.05


def test_preprocessing_fit_on_training_only():
    """Leakage test: imputer/scaler statistics must come from the fit split, not test/all data."""
    df = reg_df()
    df.loc[df.index[-60:], "a"] += 1000          # shift rows; stats differ by split
    r = train_model(df, TrainConfig("y", "regression", "Linear Regression", seed=3))
    scaler = r.pipeline.named_steps["prep"].named_transformers_["num"].named_steps["scale"]
    assert np.isclose(scaler.mean_[0], r.X_fit["a"].mean())
    assert not np.isclose(scaler.mean_[0], df["a"].mean())


def test_missing_values_and_unseen_categories():
    r = train_model(clf_df(), TrainConfig("y", "classification", "Logistic Regression"))
    new = pd.DataFrame({"a": [np.nan], "b": ["NEVER_SEEN"], "c": [0.1]})
    assert r.pipeline.predict(new)[0] in ("yes", "no")


def test_numeric_only_and_categorical_only():
    df = clf_df(nan=False)
    assert train_model(df, TrainConfig("y", "classification", "Random Forest", features=["a", "c"])).features == ["a", "c"]
    assert train_model(df, TrainConfig("y", "classification", "Decision Tree", features=["b"])).features == ["b"]


def test_trees_not_scaled_linear_scaled():
    k = split_feature_kinds(clf_df()[["a", "c"]])
    assert "scale" in build_preprocessor(k, True).transformers[0][1].named_steps
    assert "scale" not in build_preprocessor(k, False).transformers[0][1].named_steps


@pytest.mark.parametrize("kwargs", [
    dict(model_name="Linear Regression"),                       # wrong model for task
    dict(test_size=0.9), dict(cal_size=0.0), dict(seed=-1), dict(seed=True),
    dict(hyperparams={"max_depth": 10_000}), dict(hyperparams={"bogus": 1}),
    dict(hyperparams={"max_depth": "deep"}), dict(features=["y"]), dict(features=["nope"]),
])
def test_invalid_configs(kwargs):
    base = dict(target="y", task="classification", model_name="Decision Tree")
    with pytest.raises(ValidationError):
        train_model(clf_df(), TrainConfig(**{**base, **kwargs}))


def test_insufficient_data_and_bad_targets():
    with pytest.raises(ValidationError):
        train_model(clf_df(n=20), TrainConfig("y", "classification", "Decision Tree"))
    df = clf_df(); df["y"] = "same"
    with pytest.raises(ValidationError):
        train_model(df, TrainConfig("y", "classification", "Decision Tree"))
    with pytest.raises(ValidationError):
        train_model(clf_df(), TrainConfig("y", "regression", "Linear Regression"))


def test_rare_class_falls_back_with_warning():
    df = clf_df(nan=False)
    df.loc[df.index[:1], "y"] = "rare"
    r = train_model(df, TrainConfig("y", "classification", "Decision Tree"))
    assert any("Stratified" in w for w in r.warnings)


def test_imbalance_warning_and_baseline():
    df = clf_df(nan=False)
    df["y"] = np.where(np.arange(len(df)) % 10 == 0, "yes", "no")
    r = train_model(df, TrainConfig("y", "classification", "Decision Tree"))
    assert any("imbalance" in w.lower() for w in r.warnings)
    assert evaluate(r.pipeline, r.X_test, r.y_test, "classification")["majority_baseline_accuracy"] >= 0.85


def test_metrics_exact_values():
    class Fake:
        classes_ = np.array(["n", "p"])
        def predict(self, X): return np.array(["p", "p", "n", "n"])
        def predict_proba(self, X): return np.array([[.2, .8], [.4, .6], [.7, .3], [.9, .1]])
    y = pd.Series(["p", "n", "n", "n"])
    m = evaluate(Fake(), None, y, "classification")
    assert m["accuracy"] == 0.75 and m["precision"] == 0.5 and m["recall"] == 1.0
    assert np.isclose(m["f1"], 2 / 3) and m["roc_auc"] == 1.0
    assert np.isclose(m["brier"], np.mean([.04, .36, .09, .01]))


def test_undefined_metrics_when_class_missing():
    r = train_model(clf_df(), TrainConfig("y", "classification", "Decision Tree"))
    one = r.y_test == r.y_test.iloc[0]
    m = evaluate(r.pipeline, r.X_test[one], r.y_test[one], "classification")
    assert m["roc_auc"] is None and m["notes"]


def test_compare_models_uses_validation_not_test():
    t = compare_models(clf_df(), TrainConfig("y", "classification", "Decision Tree"))
    assert list(t.index) == MODELS_BY_TASK["classification"] and t["f1"].between(0, 1).all()
