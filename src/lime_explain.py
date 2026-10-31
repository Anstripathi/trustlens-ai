"""Bounded local LIME explanations for the fitted tabular sklearn pipelines."""
from __future__ import annotations

import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer

MAX_LIME_BACKGROUND_ROWS = 300
MAX_LIME_SAMPLES = 5_000


class LimeUnavailableError(ValueError):
    """The requested local surrogate explanation is not available."""


def explain_lime(result, row: pd.DataFrame, class_label=None,
                 num_samples: int = 1_000, num_features: int | None = None) -> dict:
    """Explain one input row with a deterministic local tabular surrogate."""
    features = list(result.features)
    if len(row) != 1:
        raise LimeUnavailableError("LIME explains one observation at a time.")
    if not features:
        raise LimeUnavailableError("No model features are available to explain.")
    if num_features is None:
        num_features = min(10, len(features))
    if isinstance(num_samples, bool) or not isinstance(num_samples, int) or not 100 <= num_samples <= MAX_LIME_SAMPLES:
        raise LimeUnavailableError(f"LIME sample count must be between 100 and {MAX_LIME_SAMPLES}.")
    if isinstance(num_features, bool) or not isinstance(num_features, int) or not 1 <= num_features <= min(20, len(features)):
        raise LimeUnavailableError(f"Select between 1 and {min(20, len(features))} explanation features.")
    missing = [name for name in features if name not in row.columns]
    if missing:
        raise LimeUnavailableError(f"Explanation input is missing feature(s): {missing}.")

    fit = result.X_fit.loc[:, features]
    medians = {}
    category_values = {}
    category_maps = {}
    categorical_indices = []
    categorical_names = {}
    for index, name in enumerate(features):
        if result.feature_kinds[name] == "numeric":
            numeric = pd.to_numeric(fit[name], errors="coerce")
            medians[name] = float(numeric.median()) if numeric.notna().any() else 0.0
        else:
            categorical_indices.append(index)
            values = list(pd.unique(fit[name].dropna()))
            if "missing" not in values:
                values.append("missing")
            category_values[name] = values
            category_maps[name] = {
                (type(value).__name__, repr(value)): code for code, value in enumerate(values)
            }
            categorical_names[index] = [str(value) for value in values]

    def encode(frame: pd.DataFrame) -> np.ndarray:
        out = np.empty((len(frame), len(features)), dtype=float)
        for index, name in enumerate(features):
            if result.feature_kinds[name] == "numeric":
                values = pd.to_numeric(frame[name], errors="coerce").fillna(medians[name])
                out[:, index] = values.to_numpy(dtype=float)
            else:
                mapping = category_maps[name]
                out[:, index] = [
                    mapping[("str", repr("missing"))] if pd.isna(value)
                    else mapping.get((type(value).__name__, repr(value)), mapping[("str", repr("missing"))])
                    for value in frame[name]
                ]
        return out

    def decode(values) -> pd.DataFrame:
        array = np.asarray(values, dtype=float)
        data = {}
        for index, name in enumerate(features):
            if result.feature_kinds[name] == "numeric":
                data[name] = array[:, index]
            else:
                categories = category_values[name]
                decoded = []
                for code in array[:, index]:
                    category_index = int(round(code))
                    category_index = min(max(category_index, 0), len(categories) - 1)
                    decoded.append(categories[category_index])
                data[name] = decoded
        return pd.DataFrame(data, columns=features)

    background = fit
    if len(background) > MAX_LIME_BACKGROUND_ROWS:
        background = background.sample(MAX_LIME_BACKGROUND_ROWS, random_state=result.config.seed)
    train_data = encode(background)
    instance = encode(row.loc[:, features])[0]

    if result.task == "classification":
        classes = list(result.classes or [])
        if not classes or not hasattr(result.pipeline, "predict_proba"):
            raise LimeUnavailableError("This classifier does not support probability-based LIME explanations.")
        if class_label is None:
            class_label = result.pipeline.predict(row.loc[:, features])[0]
        if class_label not in classes:
            raise LimeUnavailableError("Select a class supported by this model.")
        class_index = classes.index(class_label)
        mode = "classification"
        output_name = f"probability of class {class_label!r}"

        def predict_fn(values):
            return np.asarray(result.pipeline.predict_proba(decode(values)), dtype=float)

    elif result.task == "regression":
        class_index = 0
        mode = "regression"
        output_name = "predicted value"

        def predict_fn(values):
            return np.asarray(result.pipeline.predict(decode(values)), dtype=float)

    else:
        raise LimeUnavailableError(f"Unsupported prediction task: {result.task!r}.")

    try:
        explainer = LimeTabularExplainer(
            train_data, mode=mode, feature_names=features,
            categorical_features=categorical_indices,
            categorical_names=categorical_names,
            discretize_continuous=True,
            random_state=result.config.seed,
        )
        kwargs = {"labels": (class_index,)} if mode == "classification" else {}
        explanation = explainer.explain_instance(
            instance, predict_fn, num_features=num_features, num_samples=num_samples, **kwargs
        )
        raw_weights = explanation.local_exp[class_index]
        contributions = pd.DataFrame([
            {"feature": features[index], "value": row.iloc[0][features[index]], "weight": float(weight)}
            for index, weight in raw_weights
        ]).sort_values("weight", key=lambda values: values.abs(), ascending=False).reset_index(drop=True)
        model_prediction = predict_fn(instance.reshape(1, -1))[0]
        if mode == "classification":
            model_output = float(model_prediction[class_index])
            predicted_label = result.classes[int(np.argmax(model_prediction))]
            local_prediction = float(np.asarray(explanation.local_pred).reshape(-1)[0])
        else:
            model_output = float(model_prediction)
            predicted_label = float(model_output)
            local_prediction = float(np.asarray(explanation.local_pred).reshape(-1)[0])
    except Exception:  # noqa: BLE001 - avoid surfacing model/input details
        raise LimeUnavailableError(
            "LIME could not explain this model configuration. Check the selected row and try a lower sample count."
        ) from None

    return {
        "output_name": output_name,
        "class_label": class_label if mode == "classification" else None,
        "predicted_label": predicted_label,
        "model_output": model_output,
        "local_prediction": local_prediction,
        "local_r2": float(explanation.score),
        "contributions": contributions,
        "note": "LIME approximates the model locally around this observation. The approximation may be inaccurate outside this neighborhood and does not imply causation.",
        "num_samples": num_samples,
        "background_rows": len(train_data),
    }
