"""Bounded, constraint-aware counterfactual search for fitted tabular pipelines."""
from __future__ import annotations

import numpy as np
import pandas as pd


MAX_COUNTERFACTUAL_CANDIDATES = 200
MAX_COUNTERFACTUAL_CHANGES = 3
MAX_VALUES_PER_FEATURE = 8


class CounterfactualError(ValueError):
    """Invalid or unsupported counterfactual request."""


def _feature_candidates(result, name: str, original, allowed_range):
    fit = result.X_fit[name].dropna()
    if result.feature_kinds[name] == "numeric":
        observed = pd.to_numeric(fit, errors="coerce").dropna().to_numpy(dtype=float)
        if not len(observed):
            return []
        if allowed_range is None:
            lower, upper = float(observed.min()), float(observed.max())
        else:
            lower, upper = allowed_range
        inside = observed[(observed >= lower) & (observed <= upper)]
        if not len(inside):
            return []
        values = list(np.quantile(inside, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]))
        values.extend((lower, upper))
        reference = float(original) if pd.notna(original) else float(np.median(observed))
        candidates = sorted({float(v) for v in values if lower <= v <= upper and not np.isclose(v, reference)},
                            key=lambda v: (abs(v - reference), v))
    else:
        if allowed_range is not None:
            raise CounterfactualError(f"Feature range constraints apply only to numeric features ('{name}' is categorical).")
        categories = list(pd.unique(fit))
        candidates = [v for v in categories if not (pd.isna(v) and pd.isna(original)) and v != original]
    return candidates[:MAX_VALUES_PER_FEATURE]


def generate_counterfactuals(result, row: pd.DataFrame, *, desired_class=None,
                             target_range: tuple[float, float] | None = None,
                             immutable_features: set[str] | list[str] | tuple[str, ...] = (),
                             feature_ranges: dict[str, tuple[float, float]] | None = None,
                             max_candidates: int = 100, max_changes: int = 3) -> dict:
    """Search for nearby model-input scenarios meeting a requested model outcome.

    This is a finite heuristic search over empirical feature values; it does not
    estimate interventions or imply that changing a real-world feature causes an outcome.
    """
    features = list(result.features)
    if len(row) != 1:
        raise CounterfactualError("Counterfactual analysis requires exactly one observation.")
    missing = [name for name in features if name not in row.columns]
    if missing:
        raise CounterfactualError(f"Input is missing feature(s): {missing}.")
    if isinstance(max_candidates, bool) or not isinstance(max_candidates, int) or not 1 <= max_candidates <= MAX_COUNTERFACTUAL_CANDIDATES:
        raise CounterfactualError(f"Candidate limit must be between 1 and {MAX_COUNTERFACTUAL_CANDIDATES}.")
    if isinstance(max_changes, bool) or not isinstance(max_changes, int) or not 1 <= max_changes <= MAX_COUNTERFACTUAL_CHANGES:
        raise CounterfactualError(f"Maximum changes must be between 1 and {MAX_COUNTERFACTUAL_CHANGES}.")

    if result.task == "classification":
        classes = list(result.classes or [])
        if desired_class not in classes:
            raise CounterfactualError("Choose a target class supported by this classifier.")
        if target_range is not None:
            raise CounterfactualError("A target range is only supported for regression.")
    elif result.task == "regression":
        if target_range is None or len(target_range) != 2:
            raise CounterfactualError("Set a minimum and maximum target value for regression.")
        low, high = target_range
        if not np.isfinite([low, high]).all() or low > high:
            raise CounterfactualError("Target range must contain finite values with minimum <= maximum.")
        if desired_class is not None:
            raise CounterfactualError("A target class is only supported for classification.")
    else:
        raise CounterfactualError(f"Unsupported prediction task: {result.task!r}.")

    immutable = set(immutable_features)
    ranges = dict(feature_ranges or {})
    unknown = (immutable | set(ranges)) - set(features)
    if unknown:
        raise CounterfactualError(f"Unknown constrained feature(s): {sorted(unknown)}.")
    for name, bounds in ranges.items():
        if result.feature_kinds[name] != "numeric":
            raise CounterfactualError(f"Feature range constraints apply only to numeric features ('{name}' is categorical).")
        if len(bounds) != 2:
            raise CounterfactualError(f"Feature range for '{name}' requires two bounds.")
        low, high = bounds
        if not np.isfinite([low, high]).all() or low > high:
            raise CounterfactualError(f"Feature range for '{name}' must be finite with minimum <= maximum.")

    original = row.loc[:, features].copy()
    original_prediction = result.pipeline.predict(original)[0]
    original_output = float(original_prediction) if result.task == "regression" else None
    original_probabilities = None
    if result.task == "classification":
        original_probabilities = result.pipeline.predict_proba(original)[0]
        original_output = float(original_probabilities[list(result.classes).index(desired_class)])

    def meets(value):
        return value == desired_class if result.task == "classification" else target_range[0] <= float(value) <= target_range[1]

    if meets(original_prediction):
        return {"found": True, "already_satisfied": True, "reason": "The original prediction already meets the requested outcome.",
                "original_prediction": original_prediction, "original_output": original_output,
                "candidates": pd.DataFrame(), "evaluated": 0}

    choices = {}
    for name in features:
        if name in immutable:
            continue
        choices[name] = _feature_candidates(result, name, original.iloc[0][name], ranges.get(name))
    if not any(choices.values()):
        return {"found": False, "reason": "No mutable features have valid alternative values under the current constraints.",
                "original_prediction": original_prediction, "original_output": original_output,
                "candidates": pd.DataFrame(), "evaluated": 0}

    def value_key(value):
        return "<NaN>" if pd.isna(value) else f"{type(value).__name__}:{value!r}"

    states = [(original.iloc[0].to_dict(), tuple())]
    seen = {tuple(value_key(original.iloc[0][name]) for name in features)}
    found_rows = []
    evaluated = 0
    for depth in range(1, max_changes + 1):
        proposals = []
        for values, changed in states:
            for name in features:
                if name not in choices or name in changed:
                    continue
                for candidate in choices[name]:
                    next_values = dict(values)
                    next_values[name] = candidate
                    key = tuple(value_key(next_values[col]) for col in features)
                    if key in seen:
                        continue
                    seen.add(key)
                    proposals.append((next_values, changed + (name,)))
                    if len(proposals) >= max_candidates - evaluated:
                        break
                if len(proposals) >= max_candidates - evaluated:
                    break
            if len(proposals) >= max_candidates - evaluated:
                break
        if not proposals:
            break

        batch = pd.DataFrame([values for values, _ in proposals], columns=features)
        predictions = result.pipeline.predict(batch)
        probabilities = result.pipeline.predict_proba(batch) if result.task == "classification" else None
        evaluated += len(proposals)
        ranked = []
        for position, (values, changed) in enumerate(proposals):
            prediction = predictions[position]
            probability = (float(probabilities[position, list(result.classes).index(desired_class)])
                           if probabilities is not None else float(prediction))
            record = dict(values)
            record.update(prediction=prediction, desired_output=probability,
                          changed_features=list(changed), change_count=len(changed))
            ranked.append((record, probability))
            if meets(prediction):
                found_rows.append(record)

        if found_rows:
            break
        if evaluated >= max_candidates:
            break
        if result.task == "classification":
            ranked.sort(key=lambda item: item[1], reverse=True)
        else:
            low, high = target_range
            ranked.sort(key=lambda item: max(low - item[1], 0, item[1] - high))
        states = [(row_record, tuple(row_record["changed_features"]))
                  for row_record, _ in ranked[:min(20, len(ranked))]]

    if found_rows:
        if result.task == "classification":
            found_rows.sort(key=lambda record: (record["change_count"], -record["desired_output"]))
        else:
            low, high = target_range
            found_rows.sort(key=lambda record: (record["change_count"], max(low - record["desired_output"], 0, record["desired_output"] - high)))
        candidates = pd.DataFrame(found_rows)
        found = True
        reason = "Candidate scenarios found within the configured search limits."
    else:
        candidates = pd.DataFrame()
        found = False
        reason = "No valid counterfactual was found within the candidate and feature constraints."
    return {"found": found, "already_satisfied": False, "reason": reason,
            "original_prediction": original_prediction, "original_output": original_output,
            "candidates": candidates, "evaluated": evaluated,
            "target": desired_class if result.task == "classification" else target_range,
            "note": "These are model-input scenarios from a bounded search, not causal interventions or guarantees about real-world changes."}
