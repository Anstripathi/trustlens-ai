"""Deterministic, bounded governance orchestration and human decision records."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import uuid

from .auditor import AuditRequest, run_audit

ALLOWED_ACTIONS = {"review_model", "approve_for_local_use", "reject_model"}
MAX_AGENTS_PER_RUN = 5


@dataclass(frozen=True)
class GovernancePolicy:
    """Local decision thresholds; they route for review and never certify a model."""
    max_risk_score: float = 35.0
    require_no_failed_stages: bool = True

    def __post_init__(self):
        if isinstance(self.max_risk_score, bool) or not isinstance(self.max_risk_score, (int, float)) or not 0 <= self.max_risk_score <= 100:
            raise ValueError("Maximum risk score must be between 0 and 100.")


def run_governance(request: AuditRequest, *, agent_limit: int = MAX_AGENTS_PER_RUN) -> dict:
    """Run one deterministic audit pass; no loops, tools, shell, or autonomous changes."""
    if isinstance(agent_limit, bool) or not isinstance(agent_limit, int) or not 1 <= agent_limit <= MAX_AGENTS_PER_RUN:
        raise ValueError(f"Agent limit must be between 1 and {MAX_AGENTS_PER_RUN}.")
    audit = run_audit(request)
    return {"run_id": str(uuid.uuid4()), "created_at": datetime.now(timezone.utc).isoformat(),
            "agent_limit": agent_limit, "audit": audit,
            "orchestration": "fixed local audit stages; bounded single pass; no external tools"}


def policy_gate(audit, policy: GovernancePolicy | None = None) -> dict:
    """Return a review routing decision from actual audit evidence."""
    policy = policy or GovernancePolicy()
    risk = audit.risk
    reasons = []
    if risk is None or risk.score is None:
        reasons.append("Risk score is unavailable")
    elif risk.score > policy.max_risk_score:
        reasons.append("Risk score exceeds local review threshold")
    if policy.require_no_failed_stages and audit.failed_stages:
        reasons.append("One or more assessment stages failed")
    if risk and any(check.status.value == "not_evaluated" for check in audit.checks):
        reasons.append("Some checks were not evaluated")
    return {"decision": "review_required" if reasons else "eligible_for_human_review",
            "reasons": reasons, "policy": asdict(policy),
            "disclaimer": "Eligibility routes a case to a human; it is not approval, certification, or a deployment decision."}


@dataclass
class GovernanceDecision:
    decision_id: str
    run_id: str
    action: str
    status: str = "pending"
    reviewer: str | None = None
    decided_at: str | None = None


@dataclass
class GovernanceQueue:
    """Session-local metadata queue; identity and role claims are not authenticated."""
    records: list[GovernanceDecision] = field(default_factory=list)

    def request(self, run_id: str, action: str) -> GovernanceDecision:
        if action not in ALLOWED_ACTIONS:
            raise ValueError("Requested governance action is not allowed.")
        record = GovernanceDecision(str(uuid.uuid4()), str(run_id), action)
        self.records.insert(0, record)
        del self.records[100:]
        return record

    def decide(self, decision_id: str, reviewer: str, role: str, outcome: str) -> GovernanceDecision:
        if role not in {"reviewer", "admin"}:
            raise ValueError("Only a reviewer or admin role may record a decision.")
        reviewer = str(reviewer).strip()
        if not reviewer or len(reviewer) > 100 or outcome not in {"approved", "rejected"}:
            raise ValueError("A reviewer name and approved/rejected outcome are required.")
        match = next((r for r in self.records if r.decision_id == decision_id), None)
        if match is None or match.status != "pending":
            raise ValueError("Pending governance decision was not found.")
        match.status, match.reviewer = outcome, reviewer
        match.decided_at = datetime.now(timezone.utc).isoformat()
        return match
