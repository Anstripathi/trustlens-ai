"""Resource-bounded SHAP explanations for the app's fitted sklearn pipelines."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from .preprocessing import build_preprocessor
from .train import MODELS_BY_TASK, SCALED_MODELS, make_estimator

MAX_BACKGROUND_ROWS = 10
MAX_EXPLANATION_ROWS = 25
MAX_MASKED_MODEL_ROWS = 25_000


class ShapUnavailableError(ValueError):
    """The requested SHAP explanation cannot be computed safely or correctly."""


def explain_shap(result, rows: pd.DataFrame | None = None, class_label=None,
                 background_size: int = MAX_BACKGROUND_ROWS,
                 max_rows: int = MAX_EXPLANATION_ROWS) -> dict:
    """Explain original input features with permutation SHAP.

    One fixed scalar model output is explained: regression prediction, or a selected
    classification class probability. This model-agnostic route keeps preprocessing
    inside the fitted pipeline and avoids pretending encoded columns are original inputs.
    """
    if not hasattr(result.pipeline, "predict"):
        raise ShapUnavailableError("This model does not provide prediction support.")
    if not result.features:
        raise ShapUnavailableError("No model features are available to explain.")
    if isinstance(background_size, bool) or not isinstance(background_size, int) or not 2 <= background_size <= MAX_BACKGROUND_ROWS:
        raise ShapUnavailableError(f"Background size must be between 2 and {MAX_BACKGROUND_ROWS} rows.")
    if isinstance(max_rows, bool) or not isinstance(max_rows, int) or not 1 <= max_rows <= MAX_EXPLANATION_ROWS:
        raise ShapUnavailableError(f"Explanation sample must be between 1 and {MAX_EXPLANATION_ROWS} rows.")

    features = list(result.features)
    if result.task == "classification":
        classes = list(result.classes or [])
        if not classes:
            raise ShapUnavailableError("The classifier does not expose its classes.")
        if class_label is None:
            class_label = classes[-1]
        if class_label not in classes:
            raise ShapUnavailableError("Select a class supported by this model.")
        class_index = classes.index(class_label)
        output_name = f"probability of class {class_label!r}"
    elif result.task == "regression":
        class_index = None
        output_name = "predicted value"
    else:
        raise ShapUnavailableError(f"Unsupported prediction task: {result.task!r}.")

    if rows is None:
        rows = result.X_test
    if not isinstance(rows, pd.DataFrame):
        raise ShapUnavailableError("Explanation inputs must be a feature DataFrame.")
    missing = [name for name in features if name not in rows.columns]
    if missing:
        raise ShapUnavailableError(f"Explanation input is missing feature(s): {missing}.")
    if rows.empty:
        raise ShapUnavailableError("There are no rows available to explain.")
    rows = rows.loc[:, features]
    if len(rows) > max_rows:
        rows = rows.sample(max_rows, random_state=result.config.seed).sort_index()

    background = result.X_fit.loc[:, features]
    if len(background) > background_size:
        background = background.sample(background_size, random_state=result.config.seed).sort_index()
    if len(background) < 2:
        raise ShapUnavailableError("At least two training rows are needed as SHAP background data.")

    # SHAP's tabular masker performs arithmetic on arrays. Encode categorical
    # values as stable numeric tokens for masking, then decode before prediction.
    category_values = {}
    category_to_code = {}
    for name in features:
        if result.feature_kinds[name] == "categorical":
            observed = pd.concat([result.X_fit[name], rows[name]], ignore_index=True).dropna()
            values = list(pd.unique(observed))
            category_values[name] = values
            category_to_code[name] = {
                (type(value).__name__, repr(value)): float(index)
                for index, value in enumerate(values)
            }

    def encode(frame):
        encoded = np.empty((len(frame), len(features)), dtype=float)
        for col, name in enumerate(features):
            if result.feature_kinds[name] == "numeric":
                encoded[:, col] = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
            else:
                mapping = category_to_code[name]
                encoded[:, col] = [
                    np.nan if pd.isna(value) else mapping[(type(value).__name__, repr(value))]
                    for value in frame[name]
                ]
        return encoded

    def decode(values):
        array = np.asarray(values, dtype=float)
        frame = {}
        for col, name in enumerate(features):
            if result.feature_kinds[name] == "numeric":
                frame[name] = array[:, col]
            else:
                categories = category_values[name]
                decoded = []
                for code in array[:, col]:
                    index = int(round(code))
                    if np.isnan(code) or index < 0 or index >= len(categories):
                        decoded.append(np.nan)
                    else:
                        decoded.append(categories[index])
                frame[name] = decoded
        return pd.DataFrame(frame, columns=features)

    # Three forward/reverse permutations reduce order noise while the global
    # row budget keeps expanded masked-model inputs bounded.
    max_evals = 6 * len(features) + 1
    max_explanations = MAX_MASKED_MODEL_ROWS // (len(background) * max_evals)
    if max_explanations < 1:
        raise ShapUnavailableError(
            "This feature count exceeds the configured SHAP compute budget; reduce the feature set."
        )
    if len(rows) > max_explanations:
        rows = rows.iloc[:max_explanations]

    def model_output(values):
        frame = decode(values)
        if result.task == "regression":
            return np.asarray(result.pipeline.predict(frame), dtype=float).reshape(-1)
        return np.asarray(result.pipeline.predict_proba(frame), dtype=float)[:, class_index]

    try:
        import shap

        background_encoded = encode(background)
        rows_encoded = encode(rows)
        masker = shap.maskers.Independent(background_encoded, max_samples=background_size)
        explainer = shap.Explainer(model_output, masker, algorithm="permutation", seed=result.config.seed)
        values = explainer(rows_encoded, max_evals=max_evals, silent=True)
        shap_values = np.asarray(values.values, dtype=float)
        base_values = np.asarray(values.base_values, dtype=float).reshape(-1)
        predictions = np.asarray(
            result.pipeline.predict(rows) if result.task == "regression"
            else result.pipeline.predict_proba(rows)[:, class_index],
            dtype=float,
        ).reshape(-1)
    except Exception:  # noqa: BLE001 - avoid surfacing model/input details
        raise ShapUnavailableError(
            "SHAP could not explain this model configuration. Try a smaller feature set or another supported model."
        ) from None

    if shap_values.shape != (len(rows), len(features)):
        raise ShapUnavailableError("SHAP returned an unsupported output shape for this model.")
    if not np.isfinite(shap_values).all() or not np.isfinite(base_values).all():
        raise ShapUnavailableError("SHAP returned non-finite values; no explanation is shown.")

    importance = pd.DataFrame({
        "feature": features,
        "mean_abs_shap": np.mean(np.abs(shap_values), axis=0),
    }).sort_values("mean_abs_shap", ascending=False, kind="stable").reset_index(drop=True)
    contributions = pd.DataFrame(shap_values, columns=features, index=rows.index)
    return {
        "available": True,
        "method": "SHAP permutation explainer over original input features",
        "output_name": output_name,
        "class_label": class_label,
        "importance": importance,
        "contributions": contributions,
        "values": shap_values,
        "base_values": base_values,
        "predictions": predictions,
        "rows": rows.copy(),
        "background_rows": len(background),
        "explained_rows": len(rows),
        "max_evals_per_row": max_evals,
        "note": "Attributions describe model behavior, not causation. Permutation SHAP is approximate; features may be dependent.",
    }


def compare_shap_importance(result, rows: pd.DataFrame, class_label=None,
                            background_size: int = 5, max_rows: int = 8) -> dict:
    """Fit each compatible model family on the same fit rows and compare SHAP ranks.

    The held-out test split is used only for explanation inputs; labels are not used.
    The active model reuses its already fitted pipeline.
    """
    tables = {}
    failures = {}
    for model_name in MODELS_BY_TASK[result.task]:
        try:
            if model_name == result.config.model_name:
                context = result
            else:
                pipeline = Pipeline([
                    ("prep", build_preprocessor(result.feature_kinds, scale=model_name in SCALED_MODELS)),
                    ("model", make_estimator(result.task, model_name, {}, result.config.seed)),
                ])
                pipeline.fit(result.X_fit[result.features], result.y_fit)
                context = type("ModelExplanationContext", (), {
                    "pipeline": pipeline,
                    "config": result.config,
                    "task": result.task,
                    "features": result.features,
                    "feature_kinds": result.feature_kinds,
                    "classes": list(pipeline.classes_) if result.task == "classification" else None,
                    "X_fit": result.X_fit,
                })()
            explanation = explain_shap(
                context, rows, class_label=class_label,
                background_size=background_size, max_rows=max_rows,
            )
            tables[model_name] = explanation["importance"].set_index("feature")["mean_abs_shap"]
        except Exception:  # noqa: BLE001 - keep working models usable without echoing data
            failures[model_name] = "Unsupported model configuration or compute limit exceeded."
    if not tables:
        raise ShapUnavailableError("No compatible model could be compared: " + "; ".join(failures.values()))
    comparison = pd.DataFrame(tables).reindex(result.features)
    return {"importance": comparison, "failures": failures,
            "models": list(tables), "note": "Each model uses the same fit rows and explanation sample. SHAP magnitudes are model-output-specific and should not be treated as a production ranking."}
