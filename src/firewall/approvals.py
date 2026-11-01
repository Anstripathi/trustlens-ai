"""Metadata-only human review queue for firewall decisions that require approval."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import re
import uuid

from .schemas import Action, FirewallResult


@dataclass
class ApprovalRecord:
    approval_id: str
    requested_at: str
    direction: str
    policy_id: str
    policy_version: str
    rule_ids: tuple[str, ...]
    status: str = "pending"
    decided_at: str | None = None
    reviewer: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class ApprovalQueue:
    """Keep only decision metadata; the scanned payload is never placed in the queue."""

    def __init__(self, *, max_records: int = 100):
        if isinstance(max_records, bool) or not isinstance(max_records, int) or not 1 <= max_records <= 1_000:
            raise ValueError("max_records must be between 1 and 1,000.")
        self.max_records = max_records
        self._records: list[ApprovalRecord] = []

    def submit(self, result: FirewallResult) -> ApprovalRecord:
        if result.action != Action.APPROVAL_REQUIRED:
            raise ValueError("Only approval-required firewall results can enter the queue.")
        record = ApprovalRecord(
            approval_id=str(uuid.uuid4()), requested_at=datetime.now(timezone.utc).isoformat(),
            direction=result.direction.value, policy_id=result.policy_id, policy_version=result.policy_version,
            rule_ids=tuple(dict.fromkeys(f.rule_id for f in result.findings)),
        )
        self._records.append(record)
        self._records = self._records[-self.max_records:]
        return record

    def decide(self, approval_id: str, *, approved: bool, reviewer: str) -> ApprovalRecord:
        if not isinstance(approved, bool):
            raise ValueError("Approval decision must be boolean.")
        if not isinstance(reviewer, str) or not re.fullmatch(r"[A-Za-z0-9 ._@-]{1,64}", reviewer.strip()):
            raise ValueError("Reviewer label must be 1–64 plain-text characters.")
        record = next((item for item in self._records if item.approval_id == approval_id), None)
        if record is None:
            raise ValueError("Approval record was not found in this queue.")
        if record.status != "pending":
            raise ValueError("Approval record has already been decided.")
        record.status = "approved" if approved else "rejected"
        record.decided_at = datetime.now(timezone.utc).isoformat()
        record.reviewer = reviewer.strip()
        return record

    def list(self) -> tuple[ApprovalRecord, ...]:
        return tuple(self._records)
