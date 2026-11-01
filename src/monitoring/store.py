"""SQLite persistence for bounded aggregate-only monitoring history."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import uuid
from contextlib import contextmanager

from .engine import acknowledge_alert
from .reporting import build_report

DEFAULT_MAX_RECORDS = 100
DEFAULT_RETENTION_DAYS = 30
MAX_SNAPSHOT_BYTES = 1_000_000


class MonitoringStore:
    """Store session-scoped aggregate snapshots; never accept uploaded DataFrames."""

    def __init__(self, path: str | Path, *, max_records: int = DEFAULT_MAX_RECORDS,
                 retention_days: int = DEFAULT_RETENTION_DAYS):
        if isinstance(max_records, bool) or not 1 <= max_records <= 1_000:
            raise ValueError("max_records must be between 1 and 1,000.")
        if isinstance(retention_days, bool) or not 1 <= retention_days <= 365:
            raise ValueError("retention_days must be between 1 and 365.")
        self.path = Path(path)
        self.max_records = max_records
        self.retention_days = retention_days
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS monitor_snapshots (
                scope_hash TEXT NOT NULL,
                snapshot_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                PRIMARY KEY (scope_hash, snapshot_id)
            )""")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_monitor_scope_created ON monitor_snapshots(scope_hash, created_at)")

    def _connect(self):
        return sqlite3.connect(self.path, timeout=5)

    @contextmanager
    def _connection(self):
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _scope_hash(scope_id: str) -> str:
        try:
            normalized = str(uuid.UUID(scope_id))
        except (ValueError, TypeError, AttributeError) as exc:
            raise ValueError("Monitoring scope must be a UUID.") from exc
        return hashlib.sha256(normalized.encode("ascii")).hexdigest()

    def add(self, scope_id: str, snapshot: dict) -> None:
        scope = self._scope_hash(scope_id)
        safe = build_report([snapshot])["snapshots"][0]
        payload = json.dumps(safe, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        if len(payload.encode("utf-8")) > MAX_SNAPSHOT_BYTES:
            raise ValueError("Aggregate monitoring snapshot exceeds the local storage limit.")
        created = str(safe.get("created_at") or datetime.now(timezone.utc).isoformat())
        with self._connection() as connection:
            connection.execute("DELETE FROM monitor_snapshots WHERE created_at < ?",
                               ((datetime.now(timezone.utc) - timedelta(days=self.retention_days)).isoformat(),))
            connection.execute("INSERT OR REPLACE INTO monitor_snapshots(scope_hash,snapshot_id,created_at,payload_json) VALUES(?,?,?,?)",
                               (scope, str(safe["snapshot_id"]), created, payload))
            connection.execute("""DELETE FROM monitor_snapshots WHERE scope_hash=? AND snapshot_id NOT IN
                (SELECT snapshot_id FROM monitor_snapshots WHERE scope_hash=? ORDER BY created_at DESC LIMIT ?)""",
                               (scope, scope, self.max_records))

    def list(self, scope_id: str) -> list[dict]:
        scope = self._scope_hash(scope_id)
        with self._connection() as connection:
            rows = connection.execute("SELECT payload_json FROM monitor_snapshots WHERE scope_hash=? ORDER BY created_at DESC LIMIT ?",
                                      (scope, self.max_records)).fetchall()
        result = []
        for (payload,) in rows:
            try:
                result.append(json.loads(payload))
            except (json.JSONDecodeError, TypeError):
                continue
        return result

    def acknowledge(self, scope_id: str, snapshot_id: str, alert_id: str, reviewer: str) -> dict:
        scope = self._scope_hash(scope_id)
        with self._connection() as connection:
            row = connection.execute("SELECT payload_json FROM monitor_snapshots WHERE scope_hash=? AND snapshot_id=?",
                                     (scope, snapshot_id)).fetchone()
            if row is None:
                raise ValueError("Monitoring snapshot was not found in this session history.")
            snapshot = json.loads(row[0])
            acknowledge_alert(snapshot, alert_id, reviewer)
            payload = json.dumps(snapshot, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            connection.execute("UPDATE monitor_snapshots SET payload_json=? WHERE scope_hash=? AND snapshot_id=?",
                               (payload, scope, snapshot_id))
        return snapshot
