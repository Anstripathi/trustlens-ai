"""Dataset profiling: quality statistics, column roles and leakage heuristics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

ID_UNIQUE_RATIO = 0.95
ID_NAME_HINTS = ("id", "uuid", "guid", "index", "key")


def _looks_like_id_name(name: str) -> bool:
    n = name.lower().strip()
    return n in ID_NAME_HINTS or n.endswith("_id") or n.endswith(" id") or n.startswith("id_")


def column_roles(df: pd.DataFrame) -> dict[str, str]:
    """Roles: empty, constant, identifier, numeric, categorical (heuristic, user can override)."""
    roles = {}
    n = len(df)
    for c in df.columns:
        s = df[c].dropna()
        nun = s.nunique()
        if len(s) == 0:
            roles[c] = "empty"
        elif nun <= 1:
            roles[c] = "constant"
        else:
            numeric = is_numeric_dtype(df[c])
            integer_like = numeric and bool((s % 1 == 0).all())
            high_card = n >= 20 and nun / max(len(s), 1) >= ID_UNIQUE_RATIO
            sequential = integer_like and (s.max() - s.min() + 1) <= 1.5 * len(s)
            if high_card and (not numeric or sequential):
                roles[c] = "identifier"
            elif _looks_like_id_name(c) and nun / max(len(s), 1) > 0.5:
                roles[c] = "identifier"
            else:
                roles[c] = "numeric" if is_numeric_dtype(df[c]) else "categorical"
    return roles


def leakage_risks(df: pd.DataFrame, target: str, corr_threshold: float = 0.98) -> list[str]:
    """Heuristic leakage warnings. These are hints, not proof."""
    risks = []
    if target not in df.columns:
        return risks
    y = df[target]
    mask_y = y.notna()
    y_codes = pd.Series(pd.factorize(y[mask_y])[0], index=y[mask_y].index)
    if is_numeric_dtype(y):
        y_codes = y[mask_y].astype(float)
    for c in df.columns:
        if c == target:
            continue
        if target.lower() in c.lower() and len(target) >= 3:
            risks.append(f"'{c}' has a name containing the target name '{target}' (possible derived/leaky column).")
        s = df[c]
        m = mask_y & s.notna()
        if m.sum() < 10 or s[m].nunique() < 2:
            continue
        if is_numeric_dtype(s):
            if y_codes.loc[m[m].index].nunique() < 2:
                continue
            rho = pd.Series(s[m].astype(float)).corr(y_codes.loc[m[m].index], method="spearman")
            if pd.notna(rho) and abs(rho) >= corr_threshold:
                risks.append(f"'{c}' is almost perfectly rank-correlated with the target (|rho|={abs(rho):.3f}).")
        else:
            nun = s[m].nunique()
            if nun < 0.5 * m.sum():
                purity = (pd.crosstab(s[m], y[m]).max(axis=1).sum()) / m.sum()
                if purity >= 0.99 and y[m].nunique() > 1:
                    risks.append(f"Categories of '{c}' almost perfectly determine the target (purity={purity:.3f}).")
    return risks


def correlation_matrix(df: pd.DataFrame, max_cols: int = 30) -> pd.DataFrame | None:
    num = df.select_dtypes(include="number")
    num = num.loc[:, num.nunique() > 1]
    if num.shape[1] < 2:
        return None
    return num.iloc[:, :max_cols].corr()


def profile(df: pd.DataFrame, target: str | None = None) -> dict:
    roles = column_roles(df)
    miss = df.isna().sum()
    missing = pd.DataFrame({"missing": miss, "missing_%": (miss / len(df) * 100).round(2),
                            "unique": df.nunique(), "dtype": df.dtypes.astype(str),
                            "role": pd.Series(roles)})
    num = df.select_dtypes(include="number")
    cat = df.drop(columns=num.columns)
    cat_summary = pd.DataFrame({
        "unique": cat.nunique(),
        "top": [cat[c].mode().iloc[0] if cat[c].notna().any() else None for c in cat.columns],
        "top_freq": [int((cat[c] == cat[c].mode().iloc[0]).sum()) if cat[c].notna().any() else 0
                     for c in cat.columns],
    }) if cat.shape[1] else pd.DataFrame(columns=["unique", "top", "top_freq"])
    out = {
        "n_rows": len(df),
        "n_cols": df.shape[1],
        "duplicate_rows": int(df.duplicated().sum()),
        "missing": missing,
        "numeric_stats": num.describe().T if num.shape[1] else pd.DataFrame(),
        "categorical_summary": cat_summary,
        "constant_columns": [c for c, r in roles.items() if r in ("constant", "empty")],
        "identifier_columns": [c for c, r in roles.items() if r == "identifier"],
        "roles": roles,
        "class_distribution": None,
        "leakage_risks": [],
    }
    if target is not None and target in df.columns:
        vc = df[target].value_counts(dropna=False)
        if df[target].nunique() <= 50:
            out["class_distribution"] = pd.DataFrame({"count": vc, "share_%": (vc / len(df) * 100).round(2)})
        out["leakage_risks"] = leakage_risks(df, target)
    return out
