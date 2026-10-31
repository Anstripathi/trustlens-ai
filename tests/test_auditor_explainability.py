from helpers import clf_df
from src.auditor.explainability_audit import audit_explainability
from src.auditor.schemas import AuditRequest, CheckStatus
from src.artifact_manager import dataset_fingerprint
from src.train import TrainConfig, train_model


def test_bounded_explanation_methods_run_and_stability_is_not_inferred():
    frame = clf_df(nan=False)
    model = train_model(frame, TrainConfig("y", "classification", "Logistic Regression"))
    fingerprint = dataset_fingerprint(frame)
    checks = audit_explainability(AuditRequest(frame, "x", fingerprint, model, fingerprint))
    by_name = {check.check_id: check for check in checks}
    assert by_name["explainability.global_importance"].status is CheckStatus.PASSED
    assert by_name["explainability.shap"].status is CheckStatus.PASSED
    assert by_name["explainability.lime"].status is CheckStatus.PASSED
    assert by_name["explainability.counterfactual"].metrics["candidate_budget"] == 20
    assert by_name["explainability.stability"].status is CheckStatus.NOT_EVALUATED
    assert by_name["explainability.lime"].metrics["samples"] == 300
    assert "'yes'" not in repr(checks) and "'no'" not in repr(checks)


def test_explainability_is_not_evaluated_when_disabled_or_model_is_stale():
    frame = clf_df(nan=False)
    model = train_model(frame, TrainConfig("y", "classification", "Decision Tree"))
    fingerprint = dataset_fingerprint(frame)
    disabled = audit_explainability(AuditRequest(frame, "x", fingerprint, model, fingerprint,
                                                  run_explainability=False))
    stale = audit_explainability(AuditRequest(frame, "x", fingerprint, model, "old"))
    assert all(check.status is CheckStatus.NOT_EVALUATED for check in disabled)
    assert all(check.status is CheckStatus.NOT_EVALUATED for check in stale)
