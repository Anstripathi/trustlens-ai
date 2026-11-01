import json

import pytest

from src.governance_records import AuditTrail, PolicyRegistry, PolicySnapshot


def test_policy_snapshots_are_versioned_and_digest_stable(tmp_path):
    registry = PolicyRegistry(tmp_path)
    first = PolicySnapshot("Default", 1, 35)
    saved = registry.publish(first)
    assert saved["digest"] == first.digest
    assert registry.publish(PolicySnapshot("Default", 2, 30))["version"] == 2
    with pytest.raises(ValueError, match="Next policy version"):
        registry.publish(PolicySnapshot("Default", 4, 25))


def test_audit_trail_hash_chain_and_claimed_role_action_control(tmp_path):
    trail = AuditTrail(tmp_path)
    policy = PolicySnapshot("P", 1, 35)
    first = trail.append(actor="admin1", role="admin", action="publish_policy",
                         object_id="P:v1", policy_digest=policy.digest, outcome="published")
    second = trail.append(actor="reviewer1", role="reviewer", action="record_decision",
                          object_id="decision-1", policy_digest=policy.digest, outcome="approved")
    assert second["previous_hash"] == first["event_hash"]
    assert len(trail.events()) == 2
    with pytest.raises(ValueError, match="not authorized"):
        trail.append(actor="viewer1", role="viewer", action="publish_policy",
                     object_id="P:v2", policy_digest=policy.digest, outcome="published")


def test_audit_trail_detects_record_tampering(tmp_path):
    trail = AuditTrail(tmp_path)
    policy = PolicySnapshot("P", 1, 35)
    trail.append(actor="admin1", role="admin", action="publish_policy",
                 object_id="P:v1", policy_digest=policy.digest, outcome="published")
    row = json.loads(trail.path.read_text(encoding="utf-8"))
    row["outcome"] = "altered"
    trail.path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash is invalid"):
        trail.events()
