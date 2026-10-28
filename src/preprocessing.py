"""Leakage-safe preprocessing built from scikit-learn transformers (fit on training data only)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


def split_feature_kinds(X: pd.DataFrame) -> dict[str, str]:
    return {c: ("numeric" if is_numeric_dtype(X[c]) else "categorical") for c in X.columns}


def prepare_X(X: pd.DataFrame, kinds: dict[str, str]) -> tuple[pd.DataFrame, list[str]]:
    """Coerce columns to the kinds seen at training time. Returns (frame, warnings)."""
    X = X.copy()
    warns = []
    for c, k in kinds.items():
        if k == "numeric":
            before = X[c].notna().sum()
            X[c] = pd.to_numeric(X[c], errors="coerce")
            lost = int(before - X[c].notna().sum())
            if lost:
                warns.append(f"{lost} non-numeric value(s) in '{c}' were treated as missing.")
        else:
            s = X[c].astype(object)
            X[c] = s.where(s.notna(), np.nan)
    return X, warns


def build_preprocessor(kinds: dict[str, str], scale: bool) -> ColumnTransformer:
    num = [c for c, k in kinds.items() if k == "numeric"]
    cat = [c for c, k in kinds.items() if k == "categorical"]
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    transformers = []
    if num:
        transformers.append(("num", Pipeline(num_steps), num))
    if cat:
        transformers.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", sparse_output=False,
                                     max_categories=config.MAX_ONEHOT_CATEGORIES)),
        ]), cat))
    if not transformers:
        raise ValueError("No feature columns to preprocess.")
    return ColumnTransformer(transformers, remainder="drop")
