import numpy as np
import pandas as pd
import pytest
from helpers import clf_df
from src.data_profile import column_roles, correlation_matrix, leakage_risks, profile
from src.data_validation import ValidationError
from src.task_detection import check_target, infer_task


def test_profile_counts_exact():
    df = pd.DataFrame({"a": [1, 1, 2, np.nan], "b": ["x", "x", "y", "y"], "k": [5, 5, 5, 5]})
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    p = profile(df, "b")
    assert p["duplicate_rows"] == 2 and p["n_rows"] == 5
    assert p["missing"].loc["a", "missing"] == 1
    assert p["constant_columns"] == ["k"]
    assert p["class_distribution"].loc["x", "count"] == 3


def test_identifier_and_continuous_float_roles():
    n = 100
    df = pd.DataFrame({"row_id": np.arange(n), "name": [f"n{i}" for i in range(n)],
                       "x": np.random.default_rng(0).normal(size=n), "g": ["a", "b"] * 50})
    r = column_roles(df)
    assert r["row_id"] == "identifier" and r["name"] == "identifier"
    big = pd.DataFrame({"k": np.arange(100), "income": np.random.default_rng(1).integers(20_000, 90_000, 100) * 1.0})
    rb = column_roles(big)
    assert rb["k"] == "identifier" and rb["income"] == "numeric"   # sequential ints = id; wide-range values are not
    assert r["x"] == "numeric" and r["g"] == "categorical"


def test_leakage_detection():
    df = clf_df(nan=False)
    df["leak"] = (df.y == "yes").astype(int) * 10.0
    df["y_copy"] = df.y
    msgs = " ".join(leakage_risks(df, "y"))
    assert "'leak'" in msgs and "'y_copy'" in msgs
    assert "'c'" not in msgs


def test_correlation_matrix_none_for_single_numeric():
    assert correlation_matrix(pd.DataFrame({"a": [1, 2, 3], "b": list("xyz")})) is None


def test_infer_task():
    assert infer_task(pd.Series(["a", "b", "a"])) == "classification"
    assert infer_task(pd.Series([0, 1, 1, 0])) == "classification"
    assert infer_task(pd.Series(np.random.default_rng(0).normal(size=100))) == "regression"


def test_check_target_errors():
    with pytest.raises(ValidationError):
        check_target(pd.Series(["a"] * 50), "classification")
    with pytest.raises(ValidationError):
        check_target(pd.Series(["a", "b"] * 50), "regression")
    with pytest.raises(ValidationError):
        check_target(pd.Series([1, 2, 3]), "classification")      # too few rows
    with pytest.raises(ValidationError):
        check_target(pd.Series(range(500)).astype(str), "classification")   # too many classes
    assert len(check_target(pd.Series([1.0, 2.0] * 20 + [np.nan]), "regression")) == 40
