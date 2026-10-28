"""Safe handling of files and artifacts. The app never loads user-supplied model files."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd

from .data_validation import ValidationError

FORBIDDEN_MODEL_EXTENSIONS = {".pkl", ".pickle", ".joblib", ".pt", ".pth", ".h5", ".hdf5", ".onnx",
                              ".npy", ".npz", ".dill", ".cloudpickle", ".skops", ".bin", ".safetensors"}


def reject_model_artifact(filename: str) -> None:
    """Serialized model files can execute code on load; they are never accepted."""
    ext = Path(str(filename)).suffix.lower()
    if ext in FORBIDDEN_MODEL_EXTENSIONS:
        raise ValidationError("Serialized model files are not accepted for security reasons "
                              "(pickle/joblib can execute arbitrary code when loaded).")


def safe_join(base: Path, name: str) -> Path:
    """Join a user-influenced filename to base, refusing traversal and odd names."""
    base = Path(base).resolve()
    if not name or re.search(r"[\x00-\x1f]", name) or "/" in name or "\\" in name or ".." in name \
            or Path(name).is_absolute() or re.match(r"^[A-Za-z]:", name):
        raise ValidationError("Unsafe file name.")
    target = (base / name).resolve()
    if base != target.parent:
        raise ValidationError("Unsafe file path.")
    return target


def save_json_report(obj: dict, base: Path, name: str) -> Path:
    """Write a JSON report (trusted, app-generated content only) under base."""
    if not name.endswith(".json"):
        raise ValidationError("Reports must be .json files.")
    path = safe_join(base, name)
    Path(base).mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    return path


def dataset_fingerprint(df: pd.DataFrame) -> str:
    """Short content hash used to detect a changed dataset (invalidate stale results)."""
    h = hashlib.sha256()
    h.update(",".join(map(str, df.columns)).encode())
    h.update(str(df.shape).encode())
    h.update(pd.util.hash_pandas_object(df.astype(str), index=True).values.tobytes())
    return h.hexdigest()[:16]


def safe_error_message(exc: Exception) -> str:
    """User-facing message: validation messages pass through; anything else is generic."""
    if isinstance(exc, (ValidationError, ValueError)):
        return str(exc)
    return "An unexpected error occurred. Check your data and settings, then try again."
