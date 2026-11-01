"""Report-safe monitoring structures and configurable alert thresholds."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import uuid


@dataclass(frozen=True)
class MonitorThresholds:
    drift_fraction_warning: float = 0.10
    drift_fraction_critical: float = 0.30
    ood_fraction_warning: float = 0.10
    ood_fraction_critical: float = 0.25
    confidence_mean_min: float = 0.55
    confidence_drop_warning: float = 0.10
    accuracy_drop_warning: float = 0.05
    accuracy_drop_critical: float = 0.15
    error_increase_warning: float = 0.10
    error_increase_critical: float = 0.25
    ece_increase_warning: float = 0.05
    p95_latency_max_ms: float | None = None

    def validate(self) -> "MonitorThresholds":
        for name, value in asdict(self).items():
            if name == "p95_latency_max_ms" and value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be numeric.")
            lo, hi = (0.0, 1.0) if name != "p95_latency_max_ms" else (0.0, 86_400_000.0)
            if not lo <= float(value) <= hi:
                raise ValueError(f"{name} is outside its supported range.")
        if self.drift_fraction_warning > self.drift_fraction_critical:
            raise ValueError("Critical drift threshold must be at least the warning threshold.")
        if self.ood_fraction_warning > self.ood_fraction_critical:
            raise ValueError("Critical OOD threshold must be at least the warning threshold.")
        if self.accuracy_drop_warning > self.accuracy_drop_critical:
            raise ValueError("Critical accuracy-drop threshold must be at least the warning threshold.")
        if self.error_increase_warning > self.error_increase_critical:
            raise ValueError("Critical error-increase threshold must be at least the warning threshold.")
        return self


@dataclass
class MonitoringSnapshot:
    created_at: str
    current_rows: int
    prediction_count: int
    reference_rows: int | None
    task: str
    prediction_summary: dict
    confidence_summary: dict | None
    feature_statistics: dict
    drift_summary: dict | None
    performance: dict | None
    latency_summary: dict | None
    ood_summary: dict | None
    model_version: str | None
    dataset_versions: dict
    health_summary: str
    alerts: list[dict] = field(default_factory=list)
    snapshot_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: str = "complete"

    def to_dict(self) -> dict:
        return asdict(self)


def create_alert(severity: str, category: str, message: str, recommendation: str) -> dict:
    return {"alert_id": str(uuid.uuid4()), "created_at": datetime.now(timezone.utc).isoformat(),
            "severity": severity, "category": category, "message": message,
            "recommendation": recommendation, "acknowledged": False,
            "reviewer": None, "acknowledged_at": None}
