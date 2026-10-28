"""Explainability: permutation importance (global) and baseline-substitution attribution (local).

SHAP is NOT used in this build. Both methods here are model-agnostic, operate on the *original*
input columns (so names always align) and describe model behaviour, not causation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

GLOBAL_METHOD = ("Permutation importance on the held-out test set: the drop in score when one original "
                 "feature is shuffled. Correlated features can share/hide importance.")
LOCAL_METHOD = ("Baseline substitution: contribution = f(x) - f(x with one feature replaced by its training "
                "reference value [median / mode]). Not additive, depends on the chosen reference, and "
                "interactions are not separated. Not SHAP.")
CAUSAL_NOTE = "Attributions describe how the model behaves, not what causes the outcome in the real world."


def global_importance(result, n_repeats: int = 5, max_rows: int = 2000) -> dict:
    X, y = result.X_test, result.y_test
    if len(X) < 2:
        return {"available": False, "reason": "Too few test rows for permutation importance."}
    if len(X) > max_rows:
        X = X.sample(max_rows, random_state=result.config.seed)
        y = y.loc[X.index]
    scoring = "balanced_accuracy" if result.task == "classification" else "neg_root_mean_squared_error"
    pi = permutation_importance(result.pipeline, X, y, scoring=scoring, n_repeats=n_repeats,
                                random_state=result.config.seed, n_jobs=1)
    table = pd.DataFrame({"feature": X.columns, "importance_mean": pi.importances_mean,
                          "importance_std": pi.importances_std}).sort_values("importance_mean", ascending=False)
    return {"available": True, "method": GLOBAL_METHOD, "scoring": scoring, "table": table.reset_index(drop=True),
            "note": CAUSAL_NOTE}


def _reference_row(result) -> dict:
    ref = {}
    for c, k in result.feature_kinds.items():
        s = result.X_fit[c].dropna()
        if k == "numeric":
            ref[c] = float(s.median()) if len(s) else 0.0
        else:
            ref[c] = s.mode().iloc[0] if len(s) else "missing"
    return ref


def _output(result, X, class_idx):
    if result.task == "regression":
        return np.asarray(result.pipeline.predict(X), float)
    return result.pipeline.predict_proba(X)[:, class_idx]


def local_explanation(result, row: pd.DataFrame) -> dict:
    if len(row) != 1:
        raise ValueError("Pass exactly one row.")
    row = row[result.features]
    pred = result.pipeline.predict(row)[0]
    class_idx = 0
    if result.task == "classification":
        class_idx = 1 if len(result.classes) == 2 else result.classes.index(pred)
        target_desc = f"predicted probability of class {result.classes[class_idx]!r}"
    else:
        target_desc = "predicted value"
    ref = _reference_row(result)
    variants = [row.iloc[0].to_dict()]
    for c in result.features:
        v = row.iloc[0].to_dict()
        v[c] = ref[c]
        variants.append(v)
    V = pd.DataFrame(variants, columns=result.features)
    for c, k in result.feature_kinds.items():
        if k == "numeric":
            V[c] = pd.to_numeric(V[c], errors="coerce")
        else:
            V[c] = V[c].astype(object)
    out = _output(result, V, class_idx)
    contrib = pd.DataFrame({"feature": result.features,
                            "value": [row.iloc[0][c] for c in result.features],
                            "reference": [ref[c] for c in result.features],
                            "contribution": out[0] - out[1:]})
    contrib = contrib.reindex(contrib["contribution"].abs().sort_values(ascending=False).index).reset_index(drop=True)
    return {"prediction": pred, "output_explained": target_desc, "model_output": float(out[0]),
            "contributions": contrib, "method": LOCAL_METHOD, "note": CAUSAL_NOTE}
