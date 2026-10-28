import numpy as np
import pandas as pd


def clf_df(n=400, seed=0, nan=True):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"a": rng.normal(size=n), "b": rng.choice(list("xyz"), n), "c": rng.normal(size=n)})
    p = 1 / (1 + np.exp(-(1.5 * df.a + (df.b == "x") * 1.0)))
    df["y"] = np.where(rng.random(n) < p, "yes", "no")
    if nan:
        df.loc[rng.choice(n, 20, replace=False), "a"] = np.nan
        df.loc[rng.choice(n, 20, replace=False), "b"] = np.nan
    return df


def reg_df(n=300, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"a": rng.normal(size=n), "b": rng.choice(list("xyz"), n)})
    df["y"] = 3 * df.a + (df.b == "x") * 2 + rng.normal(0, 0.5, n)
    return df


def csv_bytes(text):
    return text.encode("utf-8")
