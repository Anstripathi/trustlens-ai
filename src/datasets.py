"""Dataset helpers: synthetic demos and an optional, failure-tolerant UCI Adult download."""
from __future__ import annotations

import urllib.request

import numpy as np
import pandas as pd

from .data_validation import LoadResult, ValidationError, load_csv_bytes

ADULT_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
ADULT_COLUMNS = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
                 "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
                 "hours-per-week", "native-country", "income"]


def load_adult(url: str = ADULT_URL, timeout: int = 20) -> tuple[LoadResult | None, str]:
    """Download UCI Adult (adult.data, 32,561 rows - NOT the full 48,842). Never raises."""
    try:
        if not url.startswith("https://"):
            return None, "Only https URLs are allowed."
        with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310 (https enforced above)
            body = r.read(30 * 1024 * 1024)
        text = ",".join(ADULT_COLUMNS) + "\n" + body.decode("utf-8", errors="replace")
        return load_csv_bytes(text.encode("utf-8"), "adult.csv"), "Loaded UCI Adult (adult.data)."
    except (OSError, ValidationError, ValueError) as exc:
        return None, f"Could not download the Adult dataset ({type(exc).__name__}). Upload a CSV instead."


def make_demo_classification(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """SYNTHETIC data for trying the app. Not real records."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": rng.integers(18, 70, n),
        "income": rng.normal(50_000, 15_000, n).round(0),
        "region": rng.choice(["north", "south", "east", "west"], n),
        "tenure_years": rng.exponential(4, n).round(1),
    })
    logit = 0.03 * (df.age - 40) + (df.income - 50_000) / 20_000 + 0.4 * (df.region == "north") - 0.1 * df.tenure_years
    df["churned"] = np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), "yes", "no")
    df.loc[rng.choice(n, 25, replace=False), "income"] = np.nan
    return df
