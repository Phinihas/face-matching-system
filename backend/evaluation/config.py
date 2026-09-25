import os
from pathlib import Path

# Paths
EVALUATION_DIR = Path(__file__).parent.resolve()
BACKEND_DIR = EVALUATION_DIR.parent
PROJECT_DIR = BACKEND_DIR.parent

DEFAULT_DATASET_DIR = PROJECT_DIR / "datasets" / "lfw-deepfunneled" / "lfw-deepfunneled"
DEFAULT_PAIRS_FILE = PROJECT_DIR / "datasets" / "pairs.csv"

# Evaluation Outputs
OUTPUT_DIR = PROJECT_DIR / "evaluation_results"
PLOTS_DIR = OUTPUT_DIR / "plots"
CSV_DIR = OUTPUT_DIR / "csv"
JSON_DIR = OUTPUT_DIR / "json"
REPORTS_DIR = OUTPUT_DIR / "reports"
LOGS_DIR = OUTPUT_DIR / "logs"
FAILURES_DIR = OUTPUT_DIR / "failures"

# Ensure directories exist
for directory in [OUTPUT_DIR, PLOTS_DIR, CSV_DIR, JSON_DIR, REPORTS_DIR, LOGS_DIR, FAILURES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Default configuration values
DEFAULT_THRESHOLD = 0.21
THRESHOLD_RANGE = [round(x * 0.01, 2) for x in range(5, 101)]  # 0.05 to 1.00
DEFAULT_WORKERS = 4
