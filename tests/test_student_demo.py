import numpy as np
import pandas as pd
import pytest
from src.student_demo import fit_student_model, make_student_data


def test_data_deterministic_and_bounded():
    a, b = make_student_data(), make_student_data()
    assert a.equals(b) and a.marks.between(0, 100).all()


def test_metrics_computed_on_heldout_and_prediction_from_model():
    sm = fit_student_model()
    assert sm.metrics["n_test"] == len(sm.test) and not set(sm.train.index) & set(sm.test.index)
    pred = sm.model.predict(sm.test[["study_hours"]])
    assert np.isclose(sm.metrics["mae"], np.mean(np.abs(sm.test.marks - pred)))
    assert 5 < sm.metrics["slope"] < 7 and sm.metrics["r2"] > 0.7
    p, w = sm.predict(5.0)
    assert np.isclose(p, sm.metrics["intercept"] + 5 * sm.metrics["slope"]) and w is None


def test_extrapolation_warning_and_invalid_input():
    sm = fit_student_model()
    assert sm.predict(20.0)[1] is not None and sm.predict(0.0)[1] is not None
    for bad in (-1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            sm.predict(bad)
