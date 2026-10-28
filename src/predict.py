"""Prediction on new rows with schema checking."""
from __future__ import annotations

import pandas as pd

from .data_validation import ValidationError
from .preprocessing import prepare_X
from .train import TrainResult


def check_schema(result: TrainResult, df: pd.DataFrame) -> list[str]:
    missing = [c for c in result.features if c not in df.columns]
    if missing:
        raise ValidationError(f"Input is missing required column(s): {missing}")
    extra = [c for c in df.columns if c not in result.features]
    return [f"Ignored extra column(s): {extra}"] if extra else []


def predict_rows(result: TrainResult, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    notes = check_schema(result, df)
    if len(df) == 0:
        raise ValidationError("No rows to predict.")
    X, warns = prepare_X(df[result.features], result.feature_kinds)
    out = pd.DataFrame({"prediction": result.pipeline.predict(X)}, index=df.index)
    if result.task == "classification":
        proba = result.pipeline.predict_proba(X)
        for i, c in enumerate(result.classes):
            out[f"p({c})"] = proba[:, i]
        out["max_probability"] = proba.max(axis=1)
    return out, notes + warns
