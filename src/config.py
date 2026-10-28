"""Central configuration and safety limits for TrustLens AI."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Upload / data limits (keep in sync with .streamlit/config.toml maxUploadSize) ---
MAX_UPLOAD_MB = 20
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
MAX_ROWS = 200_000
MAX_COLS = 200

# Strings treated as missing after whitespace stripping (case-insensitive).
# NOTE: "none" is deliberately NOT included because it can be a legitimate category.
MISSING_TOKENS = {"", "?", "na", "n/a", "nan", "null"}

# --- Modelling limits ---
MIN_ROWS_TRAIN = 30
MAX_ONEHOT_CATEGORIES = 30
MAX_CLASSES = 50
REGRESSION_MIN_UNIQUE = 21          # numeric targets with more unique values -> regression
TEST_SIZE_BOUNDS = (0.1, 0.5)
CAL_SIZE_BOUNDS = (0.1, 0.4)
SEED_BOUNDS = (0, 2**31 - 1)

HP_BOUNDS = {
    "max_depth": (1, 30),
    "n_estimators": (10, 300),
    "min_samples_leaf": (1, 50),
    "C": (0.001, 1000.0),
}

# --- Reliability ---
MIN_CAL_ROWS = 50           # minimum calibration rows for probability calibration
MIN_CAL_PER_CLASS = 10
DEFAULT_CONFIDENCE_THRESHOLD = 0.6
DEFAULT_OOD_FLAG_RATE = 0.05
DEFAULT_ALPHA = 0.1         # conformal miscoverage

# --- Drift ---
DRIFT_MIN_SAMPLES = 30
DRIFT_ALPHA = 0.05
DRIFT_MIN_KS = 0.1          # practical-significance threshold (KS statistic)
DRIFT_MIN_CRAMERS_V = 0.1   # practical-significance threshold (Cramer's V)

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
