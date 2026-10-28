"""Metric calculation from held-out predictions."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn import metrics as skm


def _classification(pipe, X, y) -> dict:
    classes = list(pipe.classes_)
    pred = pipe.predict(X)
    binary = len(classes) == 2
    avg = "binary" if binary else "macro"
    kw = {"pos_label": classes[1]} if binary else {}
    out = {
        "accuracy": skm.accuracy_score(y, pred),
        "balanced_accuracy": skm.balanced_accuracy_score(y, pred),
        "precision": skm.precision_score(y, pred, average=avg, zero_division=0, **kw),
        "recall": skm.recall_score(y, pred, average=avg, zero_division=0, **kw),
        "f1": skm.f1_score(y, pred, average=avg, zero_division=0, **kw),
        "average": f"binary (positive class = {classes[1]!r})" if binary else "macro",
        "majority_baseline_accuracy": float(pd.Series(y).value_counts(normalize=True).iloc[0]),
        "notes": [],
    }
    cm = skm.confusion_matrix(y, pred, labels=classes)
    out["confusion_matrix"] = pd.DataFrame(cm, index=[f"actual: {c}" for c in classes],
                                           columns=[f"predicted: {c}" for c in classes])
    out["classification_report"] = pd.DataFrame(
        skm.classification_report(y, pred, labels=classes, output_dict=True, zero_division=0)).T
    out["roc_auc"] = out["average_precision"] = out["brier"] = None
    present = set(pd.Series(y).unique())
    if hasattr(pipe, "predict_proba") and present >= set(classes):
        proba = pipe.predict_proba(X)
        if binary:
            yb = (pd.Series(y).values == classes[1]).astype(int)
            out["roc_auc"] = skm.roc_auc_score(yb, proba[:, 1])
            out["average_precision"] = skm.average_precision_score(yb, proba[:, 1])
            out["brier"] = skm.brier_score_loss(yb, proba[:, 1])
        else:
            out["roc_auc"] = skm.roc_auc_score(y, proba, multi_class="ovr", labels=classes)
    else:
        out["notes"].append("ROC-AUC / PR / Brier are undefined because the evaluation set lacks some classes.")
    return out


def _regression(pipe, X, y) -> dict:
    pred = pipe.predict(X)
    mse = skm.mean_squared_error(y, pred)
    y = np.asarray(y, dtype=float)
    return {
        "mae": skm.mean_absolute_error(y, pred),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": skm.r2_score(y, pred) if len(y) >= 2 else None,
        "mean_baseline_rmse": float(np.sqrt(np.mean((y - y.mean()) ** 2))),
        "notes": [],
    }


def evaluate(pipe, X, y, task: str) -> dict:
    return _classification(pipe, X, y) if task == "classification" else _regression(pipe, X, y)


METRIC_HELP = {
    "accuracy": "Share of correct predictions. Misleading when one class dominates.",
    "precision": "Of predicted positives, how many were truly positive.",
    "recall": "Of true positives, how many were found.",
    "f1": "Harmonic mean of precision and recall.",
    "roc_auc": "Ranking quality across thresholds; ignores calibration; can look good under imbalance.",
    "brier": "Mean squared error of predicted probabilities (lower is better).",
    "mae": "Average absolute error, in target units.",
    "rmse": "Root of mean squared error; penalises large errors more.",
    "r2": "Variance explained relative to predicting the mean; can be negative on held-out data.",
}
