import numpy as np
import pandas as pd
import pytest
from src.drift import benjamini_hochberg, compare_schemas, drift_report


def frames(seed=0, n=500):
    rng = np.random.default_rng(seed)
    ref = pd.DataFrame({"num_same": rng.normal(0, 1, n), "num_shift": rng.normal(0, 1, n),
                        "cat_same": rng.choice(["a", "b", "c"], n, p=[.5, .3, .2]),
                        "cat_shift": rng.choice(["a", "b", "c"], n, p=[.5, .3, .2])})
    cur = pd.DataFrame({"num_same": rng.normal(0, 1, n), "num_shift": rng.normal(1.0, 1, n),
                        "cat_same": rng.choice(["a", "b", "c"], n, p=[.5, .3, .2]),
                        "cat_shift": rng.choice(["a", "b", "c"], n, p=[.1, .3, .6])})
    return ref, cur


def status(rep):
    return rep["table"].set_index("feature")["status"].to_dict()


def test_known_drift_and_no_drift():
    s = status(drift_report(*frames()))
    assert s["num_shift"] == "drift" and s["cat_shift"] == "drift"
    assert s["num_same"] == "no_drift_detected" and s["cat_same"] == "no_drift_detected"


def test_no_false_alarms_on_identically_distributed_samples():
    ref, cur = frames(seed=5)
    cur["num_shift"] = np.random.default_rng(9).normal(0, 1, len(cur))
    cur["cat_shift"] = np.random.default_rng(9).choice(["a", "b", "c"], len(cur), p=[.5, .3, .2])
    assert not drift_report(ref, cur)["summary"]["drift_features"]


def test_statistically_significant_but_tiny_effect_not_called_drift():
    rng = np.random.default_rng(0)
    ref = pd.DataFrame({"x": rng.normal(0, 1, 50_000)}); cur = pd.DataFrame({"x": rng.normal(0.03, 1, 50_000)})
    t = drift_report(ref, cur)["table"].iloc[0]
    assert t.significant and t.status == "significant_small_effect"


def test_schema_changes_and_type_mismatch():
    ref, cur = frames()
    cur = cur.drop(columns="num_same").assign(new_col=1); cur["cat_same"] = 1.0
    s = compare_schemas(ref, cur)
    assert s["missing_in_current"] == ["num_same"] and s["new_in_current"] == ["new_col"]
    assert s["type_mismatch"] == ["cat_same"] and not s["compatible"]
    assert status(drift_report(ref, cur))["cat_same"] == "skipped"


def test_small_samples_not_tested():
    ref, cur = frames(n=500)
    rep = drift_report(ref.head(10), cur.head(10))
    assert set(status(rep).values()) == {"insufficient_data"} and rep["summary"]["tested"] == 0


def test_unseen_category_reported():
    ref, cur = frames()
    cur.loc[cur.index[:100], "cat_same"] = "NEW"
    row = drift_report(ref, cur)["table"].set_index("feature").loc["cat_same"]
    assert "NEW" in row.note and row.status == "drift"


def test_no_common_columns_and_bad_args():
    rep = drift_report(pd.DataFrame({"a": [1]}), pd.DataFrame({"b": [1]}))
    assert rep["table"].empty and not rep["schema"]["compatible"]
    with pytest.raises(ValueError):
        drift_report(*frames(), correction="x")
    with pytest.raises(ValueError):
        drift_report(*frames(), alpha=2)


def test_bh_and_bonferroni_and_missingness_and_disclaimer():
    p = np.array([0.01, 0.04, 0.03, 0.5])
    assert np.allclose(benjamini_hochberg(p), [0.04, 0.0533333, 0.0533333, 0.5])
    ref, cur = frames(); cur.loc[cur.index[:50], "num_same"] = np.nan
    rep = drift_report(ref, cur, correction="bonferroni")
    assert rep["table"].set_index("feature").loc["num_same", "missing_cur_%"] == 10.0
    assert "NOT proof" in rep["disclaimer"] and "Bonferroni" in rep["summary"]["correction"]
