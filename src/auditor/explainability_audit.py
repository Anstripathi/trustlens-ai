"""Bounded explainability checks reusing TrustLens' existing methods."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..counterfactual import generate_counterfactuals
from ..explain import global_importance
from ..lime_explain import explain_lime
from ..shap_explain import explain_shap
from .schemas import AuditCategory, AuditCheck, AuditRequest, CheckStatus

EXPLANATION_ROWS = 1
LIME_SAMPLES = 300
COUNTERFACTUAL_CANDIDATES = 20


def _check(check_id, status, summary, *, metrics=None, limitations=()):
    return AuditCheck(check_id, AuditCategory.EXPLAINABILITY, status, summary,
                      metrics=dict(metrics or {}), limitations=tuple(limitations))


def _not_evaluated(reason):
    return tuple(_check(f"explainability.{method}", CheckStatus.NOT_EVALUATED, reason)
                 for method in ("global_importance", "shap", "lime", "counterfactual", "stability"))


def audit_explainability(request: AuditRequest) -> tuple[AuditCheck, ...]:
    result = request.model
    if not request.run_explainability:
        return _not_evaluated("Explainability checks were not selected.")
    if result is None or request.model_fingerprint != request.dataset_fingerprint:
        return _not_evaluated("A model trained on the current dataset is required.")
    if not len(result.X_test):
        return _not_evaluated("The model has no held-out rows to explain.")

    row = result.X_test.iloc[:EXPLANATION_ROWS]
    checks = []
    predicted_class = result.pipeline.predict(row)[0] if result.task == "classification" else None
    try:
        importance = global_importance(result, n_repeats=5, max_rows=2_000)
        if importance.get("available"):
            table = importance["table"].head(10)
            checks.append(_check(
                "explainability.global_importance", CheckStatus.PASSED,
                "Permutation importance completed on held-out test rows.",
                metrics={"method": "permutation importance", "rows": min(len(result.X_test), 2_000),
                         "repeats": 5, "top_features": [
                             {"feature": str(item.feature), "importance_mean": float(item.importance_mean)}
                             for item in table.itertuples(index=False)]},
                limitations=("Correlated features can share or hide importance; attribution is not causality.",),
            ))
        else:
            checks.append(_check("explainability.global_importance", CheckStatus.NOT_APPLICABLE,
                                 str(importance.get("reason", "Permutation importance is unavailable."))))
    except Exception:  # noqa: BLE001 - no row/feature values in exceptions
        checks.append(_check("explainability.global_importance", CheckStatus.FAILED,
                             "Permutation importance could not be calculated."))

    shap_result = None
    try:
        shap_result = explain_shap(result, row, class_label=predicted_class,
                                   background_size=10, max_rows=1)
        importance = shap_result["importance"].head(10)
        checks.append(_check(
            "explainability.shap", CheckStatus.PASSED,
            "A bounded permutation-SHAP explanation completed for one held-out row.",
            metrics={"method": "permutation SHAP", "rows": 1, "background_rows": 10,
                     "top_features": [{"feature": str(item.feature),
                                       "mean_abs_shap": float(item.mean_abs_shap)}
                                      for item in importance.itertuples(index=False)]},
            limitations=("SHAP values are approximate, model-output-specific, and do not establish causality.",),
        ))
    except Exception:  # noqa: BLE001
        checks.append(_check("explainability.shap", CheckStatus.FAILED,
                             "SHAP could not explain this model configuration."))

    lime_result = None
    try:
        lime_result = explain_lime(result, row, class_label=predicted_class,
                                   num_samples=LIME_SAMPLES,
                                   num_features=min(5, len(result.features)))
        checks.append(_check(
            "explainability.lime", CheckStatus.PASSED,
            "A bounded LIME local approximation completed for one held-out row.",
            metrics={"method": "LIME", "rows": 1, "samples": LIME_SAMPLES,
                     "local_fit_r2": float(lime_result["local_r2"]),
                     "features": int(len(lime_result["contributions"]))},
            limitations=("LIME is a local surrogate; its fit quality may be low and it is not causal.",),
        ))
    except Exception:  # noqa: BLE001
        checks.append(_check("explainability.lime", CheckStatus.FAILED,
                             "LIME could not explain this model configuration."))

    if shap_result is not None and lime_result is not None:
        try:
            shap_direction = np.sign(shap_result["contributions"].iloc[0])
            lime_direction = lime_result["contributions"].set_index("feature")["weight"]
            paired = pd.concat([shap_direction.rename("shap"), lime_direction.rename("lime")], axis=1).dropna()
            agreement = float((np.sign(paired["shap"]) == np.sign(paired["lime"])).mean()) if not paired.empty else None
            checks.append(_check(
                "explainability.attribution_consistency", CheckStatus.PASSED,
                "SHAP and LIME contribution directions were compared for shared features.",
                metrics={"shared_features": int(len(paired)), "direction_agreement": agreement},
                limitations=("The methods have different definitions; directional agreement is descriptive, not validation.",),
            ))
        except Exception:  # noqa: BLE001
            checks.append(_check("explainability.attribution_consistency", CheckStatus.FAILED,
                                 "Explanation directions could not be compared."))
    else:
        checks.append(_check("explainability.attribution_consistency", CheckStatus.NOT_EVALUATED,
                             "Both SHAP and LIME must complete for a direction comparison."))

    try:
        if result.task == "classification":
            predicted = result.pipeline.predict(row)[0]
            desired = next((label for label in result.classes if label != predicted), None)
            if desired is None:
                checks.append(_check("explainability.counterfactual", CheckStatus.NOT_APPLICABLE,
                                     "A counterfactual target class is unavailable."))
            else:
                cf = generate_counterfactuals(result, row, desired_class=desired,
                                              max_candidates=COUNTERFACTUAL_CANDIDATES, max_changes=2)
                desired_index = list(result.classes).index(desired)
        elif result.task == "regression":
            target_values = pd.to_numeric(result.y_fit, errors="coerce").dropna()
            lower, upper = (float(value) for value in target_values.quantile([0.4, 0.6]))
            cf = generate_counterfactuals(result, row, target_range=(lower, upper),
                                          max_candidates=COUNTERFACTUAL_CANDIDATES, max_changes=2)
            desired_index = None
        else:
            raise ValueError("Unsupported model task")
        checks.append(_check(
            "explainability.counterfactual", CheckStatus.PASSED,
            "A bounded empirical counterfactual search completed.",
            metrics={"found": bool(cf.get("found")), "already_satisfied": bool(cf.get("already_satisfied", False)),
                     "candidates_evaluated": int(cf.get("evaluated", 0)),
                     "candidate_budget": COUNTERFACTUAL_CANDIDATES,
                     "target_class_ordinal": desired_index},
            limitations=("Counterfactuals are model-input scenarios, not causal interventions or real-world guarantees.",),
        ))
    except Exception:  # noqa: BLE001
        checks.append(_check("explainability.counterfactual", CheckStatus.FAILED,
                             "A bounded counterfactual search could not be completed."))

    checks.append(_check(
        "explainability.stability", CheckStatus.NOT_EVALUATED,
        "The existing application does not provide a validated explanation stability test.",
        limitations=("Stability is not inferred from a single SHAP or LIME run.",),
    ))
    return tuple(checks)
