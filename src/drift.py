"""Reference-vs-current data drift analysis with multiple-comparison correction."""
from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from scipy import stats

from . import config

DISCLAIMER = ("Drift means the input data distribution changed. It is evidence of data change, NOT proof "
              "that model accuracy has deteriorated; without labels for the current data, performance "
              "cannot be measured from these inputs alone.")


def compare_schemas(ref: pd.DataFrame, cur: pd.DataFrame) -> dict:
    common = [c for c in ref.columns if c in cur.columns]
    mism = [c for c in common if is_numeric_dtype(ref[c]) != is_numeric_dtype(cur[c])]
    return {"missing_in_current": [c for c in ref.columns if c not in cur.columns],
            "new_in_current": [c for c in cur.columns if c not in ref.columns],
            "type_mismatch": mism, "common": common,
            "compatible": bool(common) and not mism}


def benjamini_hochberg(p) -> np.ndarray:
    p = np.asarray(p, float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    adj = np.clip(np.minimum.accumulate(ranked[::-1])[::-1], 0, 1)
    out = np.empty(n)
    out[order] = adj
    return out


def _numeric(r: pd.Series, c: pd.Series, min_samples: int) -> dict:
    r = r.replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    c = c.replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if len(r) < min_samples or len(c) < min_samples:
        return {"status": "insufficient_data", "note": f"fewer than {min_samples} non-missing values"}
    ks = stats.ks_2samp(r, c)
    pooled = np.sqrt((r.var(ddof=1) + c.var(ddof=1)) / 2)
    smd = abs(c.mean() - r.mean()) / pooled if pooled > 0 else 0.0
    return {"test": "Kolmogorov-Smirnov", "statistic": float(ks.statistic), "p_value": float(ks.pvalue),
            "effect_size": float(ks.statistic), "effect_name": "KS statistic", "extra": f"std. mean diff={smd:.3f}"}


def _categorical(r: pd.Series, c: pd.Series, min_samples: int) -> dict:
    r, c = r.dropna().astype(str), c.dropna().astype(str)
    if len(r) < min_samples or len(c) < min_samples:
        return {"status": "insufficient_data", "note": f"fewer than {min_samples} non-missing values"}
    unseen = sorted(set(c) - set(r))
    rc, cc = r.value_counts(), c.value_counts()
    t = pd.concat([rc, cc], axis=1, keys=["ref", "cur"]).fillna(0)
    rare = t.sum(axis=1) < 10
    if rare.sum() > 0:
        t = pd.concat([t[~rare], t[rare].sum().to_frame("__rare__").T])
    note = (f"unseen categories in current: {unseen[:5]}" + ("..." if len(unseen) > 5 else "")) if unseen else ""
    if t.shape[0] < 2:
        return {"test": "Chi-square", "statistic": 0.0, "p_value": 1.0, "effect_size": 0.0,
                "effect_name": "Cramer's V", "note": "single category in both samples " + note}
    chi2, p, _, _ = stats.chi2_contingency(t.T.values)
    n = t.values.sum()
    v = float(np.sqrt(chi2 / (n * (min(2, t.shape[0]) - 1))))
    return {"test": "Chi-square (rare categories pooled)", "statistic": float(chi2), "p_value": float(p),
            "effect_size": v, "effect_name": "Cramer's V", "note": note}


def drift_report(ref: pd.DataFrame, cur: pd.DataFrame, alpha: float = config.DRIFT_ALPHA,
                 correction: str = "bh", min_samples: int = config.DRIFT_MIN_SAMPLES,
                 min_ks: float = config.DRIFT_MIN_KS, min_v: float = config.DRIFT_MIN_CRAMERS_V) -> dict:
    if correction not in ("bh", "bonferroni"):
        raise ValueError("correction must be 'bh' or 'bonferroni'")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    schema = compare_schemas(ref, cur)
    rows = []
    for col in schema["common"]:
        base = {"feature": col, "missing_ref_%": round(ref[col].isna().mean() * 100, 2),
                "missing_cur_%": round(cur[col].isna().mean() * 100, 2)}
        rn, cn = is_numeric_dtype(ref[col]), is_numeric_dtype(cur[col])
        if rn != cn:
            rows.append({**base, "type": "mismatch", "status": "skipped", "note": "numeric in one dataset, text in the other"})
            continue
        res = _numeric(ref[col], cur[col], min_samples) if rn else _categorical(ref[col], cur[col], min_samples)
        rows.append({**base, "type": "numeric" if rn else "categorical", **res})
    df = pd.DataFrame(rows)
    if df.empty:
        return {"schema": schema, "table": df, "summary": {"tested": 0, "note": "No comparable columns."},
                "disclaimer": DISCLAIMER}
    for col in ("p_value", "p_adj", "effect_size"):
        df[col] = df.get(col, np.nan)
    tested = df["p_value"].notna()
    if tested.any():
        p = df.loc[tested, "p_value"].values
        df.loc[tested, "p_adj"] = benjamini_hochberg(p) if correction == "bh" else np.minimum(p * len(p), 1.0)
    status = []
    for _, r in df.iterrows():
        if isinstance(r.get("status"), str):
            status.append(r["status"])
            continue
        thr = min_ks if r["type"] == "numeric" else min_v
        sig = r["p_adj"] < alpha
        status.append("drift" if sig and r["effect_size"] >= thr else
                      "significant_small_effect" if sig else "no_drift_detected")
    df["status"] = status
    df["significant"] = df["p_adj"] < alpha
    counts = df["status"].value_counts().to_dict()
    summary = {"tested": int(tested.sum()), "alpha": alpha,
               "correction": "Benjamini-Hochberg (FDR)" if correction == "bh" else "Bonferroni (FWER)",
               "counts": counts, "n_ref": len(ref), "n_cur": len(cur),
               "drift_features": df.loc[df["status"] == "drift", "feature"].tolist(),
               "practical_thresholds": {"KS": min_ks, "Cramers_V": min_v}}
    return {"schema": schema, "table": df, "summary": summary, "disclaimer": DISCLAIMER}
