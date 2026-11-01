import json

import pytest

from helpers import clf_df
from src.experiments import ExperimentRegistry, experiment_record
from src.train import TrainConfig, train_model


def test_experiment_registry_persists_metadata_only_and_is_bounded(tmp_path):
    df = clf_df()
    result = train_model(df, TrainConfig("y", "classification", "Decision Tree"))
    record = experiment_record(result, "0123456789abcdef", metrics={"f1": 0.5})
    registry = ExperimentRegistry(tmp_path)
    registry.add(record)
    assert registry.list()[0]["experiment_id"] == record["experiment_id"]
    raw = json.dumps(registry.list())
    assert "X_test" not in raw and "y_test" not in raw and "pickle" not in raw.lower()
    assert "feature_columns" in raw and "dataset_fingerprint" in raw


def test_experiment_requires_valid_fingerprint(tmp_path):
    result = train_model(clf_df(), TrainConfig("y", "classification", "Decision Tree"))
    with pytest.raises(ValueError, match="fingerprint"):
        experiment_record(result, "bad")


def test_registry_rejects_oversized_or_invalid_metadata(tmp_path):
    registry = ExperimentRegistry(tmp_path)
    with pytest.raises(ValueError, match="valid experiment"):
        registry.add({})
    registry.path.write_text("not json", encoding="utf-8")
    with pytest.raises(ValueError, match="could not be read"):
        registry.list()
