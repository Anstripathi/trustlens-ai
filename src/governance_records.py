"""Versioned policy snapshots and metadata-only hash-chained local audit log."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import uuid

from .artifact_manager import safe_join

GENESIS_HASH = "0" * 64
MAX_EVENT_BYTES = 16_000
MAX_LOG_BYTES = 20_000_000
VALID_ROLES = {"viewer", "analyst", "reviewer", "admin"}
ROLE_ACTIONS = {
    "viewer": {"view_report"},
    "analyst": {"run_assessment", "save_experiment"},
    "reviewer": {"record_decision", "view_report"},
    "admin": {"publish_policy", "record_decision", "run_assessment", "save_experiment", "view_report"},
}


@dataclass(frozen=True)
class PolicySnapshot:
    name: str
    version: int
    max_risk_score: float
    require_human_review: bool = True
    allowed_actions: tuple[str, ...] = ("review_model", "approve_for_local_use", "reject_model")

    def __post_init__(self):
        if not self.name or len(self.name) > 80 or isinstance(self.version, bool) or self.version < 1:
            raise ValueError("Policy name or version is invalid.")
        if isinstance(self.max_risk_score, bool) or not isinstance(self.max_risk_score, (int, float)) or not 0 <= self.max_risk_score <= 100:
            raise ValueError("Policy risk threshold must be between 0 and 100.")
        if not self.allowed_actions or any(a not in {"review_model", "approve_for_local_use", "reject_model"} for a in self.allowed_actions):
            raise ValueError("Policy includes an unsupported governance action.")

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def digest(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


class AuditTrail:
    """Append-only local hash chain for audit metadata; not tamper-proof storage."""

    def __init__(self, path: str | Path):
        self.directory = Path(path)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = safe_join(self.directory, "governance-audit.jsonl")

    def events(self) -> list[dict]:
        if not self.path.exists():
            return []
        if self.path.stat().st_size > MAX_LOG_BYTES:
            raise ValueError("Audit log exceeds its configured size limit.")
        out, previous = [], GENESIS_HASH
        for line_number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            try:
                event = json.loads(line)
                event_hash = event.pop("event_hash")
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(f"Audit log has an invalid record at line {line_number}.") from exc
            if event.get("previous_hash") != previous:
                raise ValueError(f"Audit log chain is invalid at line {line_number}.")
            expected = hashlib.sha256(json.dumps(event, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            if event_hash != expected:
                raise ValueError(f"Audit log hash is invalid at line {line_number}.")
            event["event_hash"] = event_hash
            previous = event_hash
            out.append(event)
        return out

    def append(self, *, actor: str, role: str, action: str, object_id: str,
               policy_digest: str, outcome: str) -> dict:
        actor, action, object_id, outcome = map(str, (actor, action, object_id, outcome))
        if role not in VALID_ROLES or not actor.strip() or len(actor) > 100:
            raise ValueError("A valid claimed role and actor are required.")
        if action not in ROLE_ACTIONS[role]:
            raise ValueError("The claimed role is not authorized for this action.")
        if not action or len(action) > 80 or not object_id or len(object_id) > 120 or not outcome or len(outcome) > 120:
            raise ValueError("Audit metadata fields exceed their limits.")
        if len(policy_digest) != 64 or any(c not in "0123456789abcdef" for c in policy_digest):
            raise ValueError("A SHA-256 policy digest is required.")
        events = self.events()
        previous = events[-1]["event_hash"] if events else GENESIS_HASH
        event = {"event_id": str(uuid.uuid4()), "timestamp": datetime.now(timezone.utc).isoformat(),
                 "actor": actor.strip(), "claimed_role": role, "action": action,
                 "object_id": object_id, "policy_digest": policy_digest,
                 "outcome": outcome, "previous_hash": previous}
        event_hash = hashlib.sha256(json.dumps(event, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        encoded = json.dumps({**event, "event_hash": event_hash}, sort_keys=True, separators=(",", ":"))
        encoded_size = len(encoded.encode())
        current_size = self.path.stat().st_size if self.path.exists() else 0
        if encoded_size > MAX_EVENT_BYTES or current_size + encoded_size + 1 > MAX_LOG_BYTES:
            raise ValueError("Audit event or local audit log exceeds its size limit.")
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(encoded + "\n")
        return {**event, "event_hash": event_hash}


class PolicyRegistry:
    """Store bounded, versioned local policy snapshots without arbitrary code."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = safe_join(self.directory, "policies.json")

    def list(self) -> list[dict]:
        if not self.path.exists():
            return []
        if self.path.stat().st_size > MAX_EVENT_BYTES * 20:
            raise ValueError("Policy registry exceeds its size limit.")
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(data, list) or any(not isinstance(p, dict) for p in data):
            raise ValueError("Policy registry has an invalid format.")
        return data

    def publish(self, policy: PolicySnapshot) -> dict:
        versions = [p for p in self.list() if p.get("name") == policy.name]
        expected = max((int(p.get("version", 0)) for p in versions), default=0) + 1
        if policy.version != expected:
            raise ValueError(f"Next policy version must be {expected}.")
        record = {**policy.to_dict(), "digest": policy.digest,
                  "published_at": datetime.now(timezone.utc).isoformat()}
        records = [*self.list(), record]
        payload = json.dumps(records, ensure_ascii=False, allow_nan=False, indent=2)
        temp = safe_join(self.directory, "policies.json.tmp")
        temp.write_text(payload, encoding="utf-8")
        temp.replace(self.path)
        return record
