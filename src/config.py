from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
MOVIELENS_100K_DIR = RAW_DATA_DIR / "ml-100k"

RANDOM_SEED = 42
TOP_K_CANDIDATES = (5, 10, 20)
DEFAULT_TOP_K = 10
RATING_SCALE = (1, 5)
HIGH_RATING_THRESHOLD = 4.0
COLD_START_MIN_INTERACTIONS = 5


def ensure_project_dirs() -> None:
    for path in [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        FIGURES_DIR,
        METRICS_DIR,
        NOTEBOOKS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
