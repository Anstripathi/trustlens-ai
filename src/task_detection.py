"""Infer and validate the supervised-learning task for a target column."""
from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from . import config
from .data_validation import ValidationError


def infer_task(y: pd.Series) -> str:
    """Suggest 'classification' or 'regression'. The user must confirm in the UI."""
    y = y.dropna()
    if len(y) == 0:
        raise ValidationError("Target has no non-missing values.")
    if not is_numeric_dtype(y):
        return "classification"
    nun = y.nunique()
    integer_like = bool((y % 1 == 0).all())
    if nun < config.REGRESSION_MIN_UNIQUE and integer_like:
        return "classification"
    return "regression"


def check_target(y: pd.Series, task: str, min_rows: int = config.MIN_ROWS_TRAIN) -> pd.Series:
    """Drop missing targets and raise informative errors for unusable targets."""
    if task not in ("classification", "regression"):
        raise ValidationError(f"Unknown task '{task}'.")
    y = y.dropna()
    if len(y) < min_rows:
        raise ValidationError(f"Only {len(y)} rows have a target value; at least {min_rows} are required.")
    if task == "regression":
        if not is_numeric_dtype(y):
            raise ValidationError("Regression needs a numeric target; the selected column contains text.")
        if not np.isfinite(y.astype(float)).all():
            raise ValidationError("Regression target contains infinite values.")
        if y.nunique() < 2:
            raise ValidationError("Regression target is constant.")
    else:
        nun = y.nunique()
        if nun < 2:
            raise ValidationError("Classification target has a single class; nothing to learn.")
        if nun > config.MAX_CLASSES:
            raise ValidationError(f"Target has {nun} classes (limit {config.MAX_CLASSES}). "
                                  "It looks continuous or like an identifier - try regression or another column.")
    return y
