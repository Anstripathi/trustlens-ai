import pytest

from src.auditor.schemas import AuditResult, RiskSummary
from src.governance import GovernancePolicy, GovernanceQueue, policy_gate, run_governance


def _audit(score=20, failed=()):
    risk = RiskSummary("low", score, {}, {}, (), 1, 1, 0, 0, {}, "local test")
    return AuditResult.create(risk=risk, failed_stages=failed)


def test_policy_gate_requires_human_review_when_score_is_high_or_evidence_is_incomplete():
    assert policy_gate(_audit(60))["decision"] == "review_required"
    assert policy_gate(_audit(20, ("model",)))["decision"] == "review_required"
    assert policy_gate(_audit(20))["decision"] == "eligible_for_human_review"


def test_policy_and_agent_limits_are_validated():
    with pytest.raises(ValueError):
        GovernancePolicy(max_risk_score=101)
    with pytest.raises(ValueError, match="Agent limit"):
        run_governance(None, agent_limit=6)


def test_governance_queue_is_allowlisted_and_requires_reviewer_claim():
    queue = GovernanceQueue()
    decision = queue.request("run-id", "approve_for_local_use")
    with pytest.raises(ValueError, match="role"):
        queue.decide(decision.decision_id, "A", "viewer", "approved")
    stored = queue.decide(decision.decision_id, "A", "reviewer", "approved")
    assert stored.status == "approved" and stored.reviewer == "A"
    with pytest.raises(ValueError, match="not allowed"):
        queue.request("run-id", "run_shell")
