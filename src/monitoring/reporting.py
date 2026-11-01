"""Privacy-conscious JSON and CSV reports for aggregate monitoring records."""
from __future__ import annotations

import csv
import io
import json
import math

import numpy as np


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def build_report(snapshots: list[dict]) -> dict:
    """Return supplied aggregate snapshots; raw uploaded rows are not accepted by this API."""
    safe_snapshots = []
    for snapshot in snapshots[-100:]:
        safe_snapshots.append(_json_safe({key: snapshot.get(key) for key in (
            "snapshot_id", "created_at", "status", "task", "current_rows", "prediction_count", "reference_rows",
            "prediction_summary", "confidence_summary", "feature_statistics", "drift_summary",
            "performance", "latency_summary", "ood_summary", "model_version",
            "dataset_versions", "health_summary", "alerts")
        }))
    return {
        "report_version": 1,
        "snapshot_count": len(safe_snapshots),
        "snapshots": safe_snapshots,
        "limitations": [
            "Metrics describe only prediction data explicitly supplied by the user; they are not live service telemetry.",
            "Unlabeled batches do not establish predictive performance or accuracy degradation.",
            "Drift and OOD signals describe input change or unusualness, not correctness.",
            "Feature names and aggregate statistics can still be sensitive and should be reviewed before sharing.",
        ],
    }


def json_report_bytes(snapshots: list[dict]) -> bytes:
    return json.dumps(build_report(snapshots), indent=2, ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def _spreadsheet_safe(value) -> str:
    text = "" if value is None else str(value)
    if text.lstrip().startswith(("=", "+", "-", "@", "\t", "\r")):
        return "'" + text
    return text


def csv_report_bytes(snapshots: list[dict]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(["record_type", "snapshot_id", "created_at", "task", "status", "current_rows",
                     "prediction_count", "reference_rows", "metric", "value", "alert_id", "alert_severity",
                     "alert_category", "alert_acknowledged", "reviewer"])
    for snapshot in build_report(snapshots)["snapshots"]:
        for feature, statistics in (snapshot.get("feature_statistics") or {}).items():
            writer.writerow(["feature", _spreadsheet_safe(snapshot.get("snapshot_id")),
                             _spreadsheet_safe(snapshot.get("created_at")), _spreadsheet_safe(snapshot.get("task")),
                             _spreadsheet_safe(snapshot.get("status")), snapshot.get("current_rows"),
                             snapshot.get("prediction_count"), snapshot.get("reference_rows"), _spreadsheet_safe(feature),
                             _spreadsheet_safe(json.dumps(statistics, ensure_ascii=False, sort_keys=True)),
                             "", "", "", "", ""])
        for section in ("performance", "confidence_summary", "latency_summary", "ood_summary"):
            values = snapshot.get(section)
            if not values:
                continue
            flat = json.dumps(values, ensure_ascii=False, sort_keys=True)
            writer.writerow([_spreadsheet_safe(section), _spreadsheet_safe(snapshot.get("snapshot_id")),
                             _spreadsheet_safe(snapshot.get("created_at")), _spreadsheet_safe(snapshot.get("task")),
                             _spreadsheet_safe(snapshot.get("status")), snapshot.get("current_rows"),
                             snapshot.get("prediction_count"), snapshot.get("reference_rows"), _spreadsheet_safe(section), _spreadsheet_safe(flat),
                             "", "", "", "", ""])
        for alert in snapshot.get("alerts") or []:
            writer.writerow(["alert", _spreadsheet_safe(snapshot.get("snapshot_id")),
                             _spreadsheet_safe(alert.get("created_at")), _spreadsheet_safe(snapshot.get("task")),
                             _spreadsheet_safe(snapshot.get("status")), snapshot.get("current_rows"),
                             snapshot.get("prediction_count"), snapshot.get("reference_rows"), "", "", _spreadsheet_safe(alert.get("alert_id")),
                             _spreadsheet_safe(alert.get("severity")), _spreadsheet_safe(alert.get("category")),
                             _spreadsheet_safe(alert.get("acknowledged")), _spreadsheet_safe(alert.get("reviewer"))])
    return output.getvalue().encode("utf-8")
