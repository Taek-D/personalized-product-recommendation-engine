from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data import dataset_overview, load_bundle
from src.models import baseline_summary


def main() -> None:
    bundle = load_bundle(download_if_missing=False)
    overview = dataset_overview(bundle)
    baselines = baseline_summary(bundle.ratings)

    print('=== DATASET OVERVIEW ===')
    for key, value in overview.items():
        print(f'{key}: {value}')

    print('\n=== BASELINE SUMMARY ===')
    for key, value in baselines.items():
        print(f'{key}: {value}')


if __name__ == '__main__':
    main()
