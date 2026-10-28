"""Safe CSV loading and validation. Uploaded content is only ever parsed as data."""
from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field

import pandas as pd

from . import config


class ValidationError(Exception):
    """A user-facing, actionable validation problem (safe to display)."""


@dataclass
class LoadResult:
    raw: pd.DataFrame          # everything as stripped-nothing strings, exactly as parsed
    clean: pd.DataFrame        # missing tokens -> NaN, whitespace stripped, numeric inferred
    notes: list[str] = field(default_factory=list)


def sanitize_column_names(columns) -> tuple[list[str], list[str]]:
    """Strip control chars/whitespace, fill blanks, de-duplicate. Returns (names, notes)."""
    notes, out, seen = [], [], {}
    for i, c in enumerate(columns):
        name = re.sub(r"[\x00-\x1f\x7f]", "", str(c)).strip()
        if not name or name.lower().startswith("unnamed:"):
            name = f"unnamed_{i}"
            notes.append(f"Column {i} had no name and was renamed '{name}'.")
        base, k = name, seen.get(name, 0)
        if k:
            name = f"{base}_{k}"
            notes.append(f"Duplicate column name '{base}' renamed '{name}'.")
        seen[base] = k + 1
        out.append(name)
    return out, notes


def _decode(data: bytes, notes: list[str]) -> str:
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        notes.append("File is not valid UTF-8; decoded as latin-1. Non-ASCII characters may be wrong.")
        return data.decode("latin-1")


def clean_dataframe(raw: pd.DataFrame) -> pd.DataFrame:
    """Explicit cleaning: strip whitespace, map missing tokens to NaN, infer numeric columns.

    A column becomes numeric only if *every* non-missing value parses as a number;
    otherwise it stays as text so no value is silently coerced to NaN.
    """
    out = {}
    for c in raw.columns:
        s = raw[c].astype(str).str.strip()
        missing = s.str.lower().isin(config.MISSING_TOKENS)
        s = s.mask(missing)
        non_missing = s.notna().sum()
        num = pd.to_numeric(s, errors="coerce")
        if non_missing > 0 and num.notna().sum() == non_missing:
            out[c] = num
        else:
            out[c] = s.astype(object).where(s.notna(), float("nan"))
    return pd.DataFrame(out)


def _parse_strict(text: str) -> pd.DataFrame:
    """Parse with the csv module; every row must have exactly as many fields as the header.

    (pandas silently turns extra leading fields into an index, so it is not used for parsing.)
    """
    reader = csv.reader(io.StringIO(text))
    header, rows = None, []
    try:
        for row in reader:
            if not row:                      # blank line
                continue
            if header is None:
                header = row
                continue
            if len(row) != len(header):
                raise ValidationError(f"Malformed CSV: line {reader.line_num} has {len(row)} fields "
                                      f"but the header has {len(header)}.")
            rows.append(row)
            if len(rows) > config.MAX_ROWS:
                raise ValidationError(f"Too many rows (limit {config.MAX_ROWS:,}).")
    except csv.Error as exc:
        raise ValidationError(f"The CSV is malformed and could not be parsed: {str(exc)[:200]}") from None
    if header is None:
        raise ValidationError("The file contains no columns or rows.")
    return pd.DataFrame(rows, columns=header, dtype=str)


def load_csv_bytes(data: bytes, filename: str = "upload.csv",
                   max_bytes: int = config.MAX_UPLOAD_BYTES) -> LoadResult:
    if not str(filename).lower().endswith(".csv"):
        raise ValidationError("Only .csv files are supported.")
    if data is None or len(data) == 0:
        raise ValidationError("The file is empty.")
    if len(data) > max_bytes:
        raise ValidationError(f"File is larger than the {max_bytes // (1024 * 1024)} MB limit.")
    if b"\x00" in data[:4096]:
        raise ValidationError("File looks binary, not a text CSV.")
    notes: list[str] = []
    text = _decode(data, notes)
    if not text.strip():
        raise ValidationError("The file contains no data.")
    raw = _parse_strict(text)
    return validate_loaded(raw, notes)


def validate_loaded(raw: pd.DataFrame, notes: list[str] | None = None) -> LoadResult:
    notes = list(notes or [])
    if raw.shape[0] == 0:
        raise ValidationError("The CSV has a header but no data rows.")
    if raw.shape[0] > config.MAX_ROWS:
        raise ValidationError(f"Too many rows (limit {config.MAX_ROWS:,}).")
    if raw.shape[1] > config.MAX_COLS:
        raise ValidationError(f"Too many columns (limit {config.MAX_COLS}).")
    if raw.shape[1] < 2:
        raise ValidationError("The dataset needs at least 2 columns (check the delimiter).")
    names, name_notes = sanitize_column_names(raw.columns)
    raw = raw.copy()
    raw.columns = names
    notes += name_notes
    clean = clean_dataframe(raw)
    notes.append("Cleaning: whitespace stripped; values "
                 f"{sorted(config.MISSING_TOKENS - {''})} and empty strings treated as missing; "
                 "columns where all non-missing values are numeric were converted to numbers.")
    return LoadResult(raw=raw, clean=clean, notes=notes)


def validate_target(df: pd.DataFrame, target: str) -> None:
    if target not in df.columns:
        raise ValidationError(f"Target column '{target}' does not exist.")
    if df[target].notna().sum() == 0:
        raise ValidationError(f"Target column '{target}' has no non-missing values.")
