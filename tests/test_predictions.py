import pandas as pd
import pytest
from helpers import clf_df, reg_df
from src.data_validation import ValidationError
from src.predict import predict_rows
from src.train import TrainConfig, train_model


def test_predict_shape_and_probabilities():
    r = train_model(clf_df(), TrainConfig("y", "classification", "Random Forest"))
    out, _ = predict_rows(r, r.X_test)
    assert len(out) == len(r.X_test)
    assert abs(out[["p(no)", "p(yes)"]].sum(axis=1) - 1).max() < 1e-9
    assert (out.max_probability >= 0.5).all()


def test_schema_mismatch_and_extras_and_coercion():
    r = train_model(reg_df(), TrainConfig("y", "regression", "Linear Regression"))
    with pytest.raises(ValidationError):
        predict_rows(r, pd.DataFrame({"a": [1.0]}))
    out, notes = predict_rows(r, pd.DataFrame({"a": ["bad"], "b": ["x"], "zzz": [1]}))
    assert len(out) == 1 and any("Ignored" in n for n in notes) and any("non-numeric" in n for n in notes)
    with pytest.raises(ValidationError):
        predict_rows(r, pd.DataFrame({"a": [], "b": []}))
