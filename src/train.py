"""Model training with leakage-safe splits: fit / calibration(validation) / test."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from . import config
from .data_profile import column_roles
from .data_validation import ValidationError
from .preprocessing import build_preprocessor, prepare_X, split_feature_kinds
from .task_detection import check_target

MODELS_BY_TASK = {
    "classification": ["Logistic Regression", "Decision Tree", "Random Forest"],
    "regression": ["Linear Regression", "Decision Tree", "Random Forest"],
}
SCALED_MODELS = {"Logistic Regression", "Linear Regression"}
ALLOWED_HP = {
    "Logistic Regression": {"C"},
    "Linear Regression": set(),
    "Decision Tree": {"max_depth", "min_samples_leaf"},
    "Random Forest": {"n_estimators", "max_depth", "min_samples_leaf"},
}


@dataclass
class TrainConfig:
    target: str
    task: str
    model_name: str
    features: list[str] | None = None
    test_size: float = 0.2
    cal_size: float = 0.2
    seed: int = 42
    hyperparams: dict = field(default_factory=dict)


@dataclass
class TrainResult:
    pipeline: Pipeline
    config: TrainConfig
    task: str
    features: list[str]
    feature_kinds: dict
    classes: list | None
    X_fit: pd.DataFrame
    y_fit: pd.Series
    X_cal: pd.DataFrame
    y_cal: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    warnings: list[str]
    split_info: dict


def validate_config(cfg: TrainConfig) -> None:
    if cfg.task not in MODELS_BY_TASK:
        raise ValidationError(f"Unknown task '{cfg.task}'.")
    if cfg.model_name not in MODELS_BY_TASK[cfg.task]:
        raise ValidationError(f"'{cfg.model_name}' is not available for {cfg.task}.")
    for name, val, (lo, hi) in (("test_size", cfg.test_size, config.TEST_SIZE_BOUNDS),
                                ("cal_size", cfg.cal_size, config.CAL_SIZE_BOUNDS)):
        if not isinstance(val, (int, float)) or not lo <= val <= hi:
            raise ValidationError(f"{name} must be between {lo} and {hi}.")
    lo, hi = config.SEED_BOUNDS
    if not isinstance(cfg.seed, (int, np.integer)) or isinstance(cfg.seed, bool) or not lo <= cfg.seed <= hi:
        raise ValidationError("Seed must be an integer between 0 and 2^31-1.")
    for k, v in cfg.hyperparams.items():
        if k not in ALLOWED_HP[cfg.model_name]:
            raise ValidationError(f"Hyperparameter '{k}' is not supported for {cfg.model_name}.")
        if isinstance(v, bool) or not isinstance(v, (int, float, np.integer, np.floating)):
            raise ValidationError(f"Hyperparameter '{k}' must be numeric.")
        lo, hi = config.HP_BOUNDS[k]
        if not lo <= v <= hi:
            raise ValidationError(f"Hyperparameter '{k}' must be between {lo} and {hi}.")


def make_estimator(task: str, name: str, hp: dict, seed: int):
    hp = dict(hp)
    for k in ("max_depth", "min_samples_leaf", "n_estimators"):
        if k in hp:
            hp[k] = int(hp[k])
    if task == "classification":
        if name == "Logistic Regression":
            return LogisticRegression(C=hp.get("C", 1.0), max_iter=1000, random_state=seed)
        if name == "Decision Tree":
            return DecisionTreeClassifier(random_state=seed, **hp)
        return RandomForestClassifier(random_state=seed, n_jobs=1, **{"n_estimators": 100, **hp})
    if name == "Linear Regression":
        return LinearRegression()
    if name == "Decision Tree":
        return DecisionTreeRegressor(random_state=seed, **hp)
    return RandomForestRegressor(random_state=seed, n_jobs=1, **{"n_estimators": 100, **hp})


def _split(X, y, size, seed, stratify, warns, label):
    strat = None
    if stratify:
        vc = y.value_counts()
        n_part = int(np.ceil(len(y) * size))
        if vc.min() >= 2 and n_part >= len(vc):
            strat = y
        else:
            warns.append(f"Stratified {label} split not possible (a class is too rare); used a random split.")
    return train_test_split(X, y, test_size=size, random_state=seed, stratify=strat)


def train_model(df: pd.DataFrame, cfg: TrainConfig) -> TrainResult:
    validate_config(cfg)
    if cfg.target not in df.columns:
        raise ValidationError(f"Target column '{cfg.target}' does not exist.")
    feats = list(cfg.features) if cfg.features else [c for c in df.columns if c != cfg.target]
    if cfg.target in feats:
        raise ValidationError("The target column cannot also be used as a feature.")
    missing = [f for f in feats if f not in df.columns]
    if missing:
        raise ValidationError(f"Unknown feature column(s): {missing}")
    if not feats:
        raise ValidationError("Select at least one feature.")

    y_all = check_target(df[cfg.target], cfg.task)
    data = df.loc[y_all.index]
    X, _ = prepare_X(data[feats], split_feature_kinds(data[feats]))
    kinds = split_feature_kinds(X)
    y = y_all
    warns: list[str] = []
    clf = cfg.task == "classification"

    X_tv, X_test, y_tv, y_test = _split(X, y, cfg.test_size, cfg.seed, clf, warns, "test")
    X_fit, X_cal, y_fit, y_cal = _split(X_tv, y_tv, cfg.cal_size, cfg.seed, clf, warns, "calibration")
    if len(X_fit) < 10:
        raise ValidationError(f"Only {len(X_fit)} rows remain for fitting; provide more data or reduce split sizes.")
    if clf and y_fit.nunique() < 2:
        raise ValidationError("The training split contains a single class; provide more data.")

    pre = build_preprocessor(kinds, scale=cfg.model_name in SCALED_MODELS)
    pipe = Pipeline([("prep", pre), ("model", make_estimator(cfg.task, cfg.model_name, cfg.hyperparams, cfg.seed))])
    pipe.fit(X_fit, y_fit)

    if len(df) < 200:
        warns.append(f"Small dataset ({len(df)} rows): metrics will be noisy and calibration/intervals may be unavailable.")
    if X.isna().any().any():
        warns.append("Missing values were imputed (median / constant 'missing'); imputers were fit on the fit split only.")
    if clf:
        shares = y.value_counts(normalize=True)
        if shares.iloc[0] > 0.75:
            warns.append(f"Class imbalance: majority class is {shares.iloc[0]:.0%}; accuracy can be misleading.")
        if y_test.nunique() < y.nunique():
            warns.append("Not every class appears in the test split; some metrics are undefined.")
    roles_hint = [c for c, r in column_roles(data[feats]).items() if r == "identifier"]
    if roles_hint:
        warns.append(f"Possible identifier column(s) used as features: {roles_hint}. They can cause overfitting.")

    info = {"n_rows_used": len(X), "n_fit": len(X_fit), "n_cal": len(X_cal), "n_test": len(X_test),
            "stratified": clf and not any("Stratified" in w for w in warns), "seed": cfg.seed,
            "strategy": "hold-out: fit / calibration(validation) / test; test set is used only for final reporting"}
    return TrainResult(pipe, cfg, cfg.task, feats, kinds,
                       list(pipe.classes_) if clf else None,
                       X_fit, y_fit, X_cal, y_cal, X_test, y_test, warns, info)


def compare_models(df: pd.DataFrame, cfg: TrainConfig) -> pd.DataFrame:
    """Compare all candidate models on the *validation (calibration) split* with identical splits.

    The test set is intentionally not used, so choosing a model from this table does not leak test labels.
    """
    from .evaluate import evaluate  # local import to avoid cycle
    from time import perf_counter

    rows = []
    for name in MODELS_BY_TASK[cfg.task]:
        c = TrainConfig(**{**cfg.__dict__, "model_name": name, "hyperparams": {}})
        train_start = perf_counter()
        res = train_model(df, c)
        fit_seconds = perf_counter() - train_start
        predict_start = perf_counter()
        res.pipeline.predict(res.X_cal)
        predict_seconds = perf_counter() - predict_start
        m = evaluate(res.pipeline, res.X_cal, res.y_cal, res.task)
        key = ("accuracy", "precision", "recall", "f1", "roc_auc", "confusion_matrix") \
            if res.task == "classification" else ("mae", "mse", "rmse", "r2")
        estimator = res.pipeline.named_steps["model"]
        if hasattr(estimator, "estimators_"):
            complexity = int(sum(tree.tree_.node_count for tree in estimator.estimators_))
            complexity_unit = "tree nodes"
        elif hasattr(estimator, "tree_"):
            complexity = int(estimator.tree_.node_count)
            complexity_unit = "tree nodes"
        elif hasattr(estimator, "coef_"):
            complexity = int(np.count_nonzero(estimator.coef_))
            complexity_unit = "non-zero coefficients"
        else:
            complexity, complexity_unit = None, "unavailable"
        rows.append({"model": name, **{k: m[k] for k in key},
                     "fit_seconds": fit_seconds,
                     "predict_ms_per_row": predict_seconds * 1000 / max(1, len(res.X_cal)),
                     "complexity": complexity, "complexity_unit": complexity_unit})
    return pd.DataFrame(rows).set_index("model")
