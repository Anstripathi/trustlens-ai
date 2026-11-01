"""Metadata-only experiment and model version registry.

Model objects are deliberately never serialized. A version is a reproducible
training recipe tied to a dataset fingerprint, not a loadable binary artifact.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

from .artifact_manager import safe_join

MAX_RECORDS = 500
MAX_REGISTRY_BYTES = 2_000_000


def experiment_record(result, dataset_fp: str, *, metrics=None, benchmark=None) -> dict:
    """Create a privacy-conscious experiment record from an existing training result."""
    if not isinstance(dataset_fp, str) or len(dataset_fp) != 16 or any(c not in "0123456789abcdef" for c in dataset_fp):
        raise ValueError("A valid dataset fingerprint is required.")
    cfg = result.config
    return {
        "experiment_id": str(uuid.uuid4()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_fingerprint": dataset_fp,
        "task": result.task,
        "model": cfg.model_name,
        "target_column": cfg.target,
        "feature_columns": list(result.features),
        "seed": int(cfg.seed),
        "test_fraction": float(cfg.test_size),
        "calibration_fraction": float(cfg.cal_size),
        "split": {k: int(v) if k.startswith("n_") else v for k, v in result.split_info.items()},
        "hyperparameters": dict(cfg.hyperparams),
        "held_out_metrics": metrics or {},
        "benchmark": benchmark or {},
        "recipe": "TrainConfig metadata only; no fitted model or dataset rows are stored.",
    }


class ExperimentRegistry:
    """Bounded local JSON metadata registry. Not an access-controlled service."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = safe_join(self.directory, "experiments.json")

    def list(self) -> list[dict]:
        if not self.path.exists():
            return []
        if self.path.stat().st_size > MAX_REGISTRY_BYTES:
            raise ValueError("Experiment registry exceeds its size limit.")
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise ValueError("Experiment registry could not be read safely.") from exc
        if not isinstance(data, list) or any(not isinstance(item, dict) for item in data):
            raise ValueError("Experiment registry has an invalid format.")
        return data[:MAX_RECORDS]

    def add(self, record: dict) -> None:
        if not isinstance(record, dict) or not record.get("experiment_id"):
            raise ValueError("A valid experiment record is required.")
        records = [record, *self.list()]
        payload = json.dumps(records[:MAX_RECORDS], ensure_ascii=False, allow_nan=False, indent=2)
        if len(payload.encode("utf-8")) > MAX_REGISTRY_BYTES:
            raise ValueError("Experiment registry exceeds its size limit.")
        temp = safe_join(self.directory, "experiments.json.tmp")
        temp.write_text(payload, encoding="utf-8")
        temp.replace(self.path)
