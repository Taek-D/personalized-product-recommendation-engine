from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.request import urlretrieve
from zipfile import ZipFile

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    COLD_START_MIN_INTERACTIONS,
    MOVIELENS_100K_DIR,
    RAW_DATA_DIR,
    RANDOM_SEED,
)

MOVIELENS_100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"

ITEM_COLUMNS = [
    "item_id",
    "title",
    "release_date",
    "video_release_date",
    "imdb_url",
    "unknown",
    "action",
    "adventure",
    "animation",
    "childrens",
    "comedy",
    "crime",
    "documentary",
    "drama",
    "fantasy",
    "film_noir",
    "horror",
    "musical",
    "mystery",
    "romance",
    "sci_fi",
    "thriller",
    "war",
    "western",
]

USER_COLUMNS = ["user_id", "age", "gender", "occupation", "zip_code"]
RATING_COLUMNS = ["user_id", "item_id", "rating", "timestamp"]


@dataclass(slots=True)
class MovieLens100KBundle:
    ratings: pd.DataFrame
    items: pd.DataFrame
    users: pd.DataFrame
    genres: pd.DataFrame


class MovieLensDataError(FileNotFoundError):
    """Raised when the expected MovieLens dataset files are missing."""


def download_movielens_100k(
    destination_dir: Path | None = None,
    force: bool = False,
) -> Path:
    """Download and extract the official MovieLens 100K dataset if needed."""
    destination_dir = destination_dir or RAW_DATA_DIR
    destination_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = destination_dir / "ml-100k"
    archive_path = destination_dir / "ml-100k.zip"

    if dataset_dir.exists() and not force:
        return dataset_dir

    if force and dataset_dir.exists():
        for path in sorted(dataset_dir.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        dataset_dir.rmdir()

    urlretrieve(MOVIELENS_100K_URL, archive_path)
    with ZipFile(archive_path) as zip_file:
        zip_file.extractall(destination_dir)

    return dataset_dir


def ensure_movielens_100k(download_if_missing: bool = False) -> Path:
    if MOVIELENS_100K_DIR.exists():
        return MOVIELENS_100K_DIR
    if download_if_missing:
        return download_movielens_100k()
    raise MovieLensDataError(
        f"MovieLens 100K dataset not found at {MOVIELENS_100K_DIR}. "
        "Place the extracted files under data/raw/ml-100k/ or call with download_if_missing=True."
    )


def load_ratings(data_dir: Path | None = None) -> pd.DataFrame:
    data_dir = data_dir or ensure_movielens_100k()
    ratings = pd.read_csv(
        data_dir / "u.data",
        sep="\t",
        names=RATING_COLUMNS,
        engine="python",
    )
    ratings["rated_at"] = pd.to_datetime(ratings["timestamp"], unit="s")
    return ratings.sort_values(["timestamp", "user_id", "item_id"]).reset_index(drop=True)


def load_items(data_dir: Path | None = None) -> pd.DataFrame:
    data_dir = data_dir or ensure_movielens_100k()
    items = pd.read_csv(
        data_dir / "u.item",
        sep="|",
        names=ITEM_COLUMNS,
        encoding="latin-1",
        engine="python",
    )
    items["release_date"] = pd.to_datetime(items["release_date"], format="%d-%b-%Y", errors="coerce")
    items["release_year"] = items["release_date"].dt.year
    items["release_decade"] = (items["release_year"] // 10 * 10).astype("Int64")
    genre_columns = genre_column_names()
    items["genre_count"] = items[genre_columns].sum(axis=1)
    items["genres"] = items[genre_columns].apply(
        lambda row: [genre for genre, flag in row.items() if flag == 1],
        axis=1,
    )
    return items


def load_users(data_dir: Path | None = None) -> pd.DataFrame:
    data_dir = data_dir or ensure_movielens_100k()
    return pd.read_csv(
        data_dir / "u.user",
        sep="|",
        names=USER_COLUMNS,
        engine="python",
    )


def load_genres(data_dir: Path | None = None) -> pd.DataFrame:
    data_dir = data_dir or ensure_movielens_100k()
    genres = pd.read_csv(
        data_dir / "u.genre",
        sep="|",
        names=["genre", "genre_id"],
        engine="python",
    )
    return genres.dropna(subset=["genre_id"]).reset_index(drop=True)


def load_bundle(
    data_dir: Path | None = None,
    download_if_missing: bool = False,
) -> MovieLens100KBundle:
    data_dir = data_dir or ensure_movielens_100k(download_if_missing=download_if_missing)
    return MovieLens100KBundle(
        ratings=load_ratings(data_dir),
        items=load_items(data_dir),
        users=load_users(data_dir),
        genres=load_genres(data_dir),
    )


def genre_column_names() -> list[str]:
    return ITEM_COLUMNS[5:]


def create_user_item_matrix(
    ratings: pd.DataFrame,
    value_column: str = "rating",
    fill_value: float = 0.0,
) -> pd.DataFrame:
    return ratings.pivot_table(
        index="user_id",
        columns="item_id",
        values=value_column,
        fill_value=fill_value,
    )


def create_implicit_feedback(
    ratings: pd.DataFrame,
    threshold: float | None = None,
) -> pd.DataFrame:
    implicit = ratings[["user_id", "item_id", "timestamp", "rated_at"]].copy()
    if threshold is None:
        implicit["interaction"] = 1
    else:
        implicit["interaction"] = (ratings["rating"] >= threshold).astype(int)
    return implicit


def interaction_counts(ratings: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    user_counts = ratings.groupby("user_id")["item_id"].count().rename("rating_count")
    item_counts = ratings.groupby("item_id")["user_id"].count().rename("rating_count")
    return user_counts.sort_values(ascending=False), item_counts.sort_values(ascending=False)


def identify_cold_start_entities(
    ratings: pd.DataFrame,
    min_interactions: int = COLD_START_MIN_INTERACTIONS,
) -> dict[str, pd.Index]:
    user_counts, item_counts = interaction_counts(ratings)
    return {
        "cold_start_users": user_counts[user_counts < min_interactions].index,
        "cold_start_items": item_counts[item_counts < min_interactions].index,
    }


def calculate_sparsity(ratings: pd.DataFrame) -> float:
    n_users = ratings["user_id"].nunique()
    n_items = ratings["item_id"].nunique()
    observed = len(ratings)
    total_possible = n_users * n_items
    return 1.0 - (observed / total_possible)


def random_train_test_split(
    ratings: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(
        ratings,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def time_based_train_test_split(
    ratings: pd.DataFrame,
    test_ratio: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sorted_ratings = ratings.sort_values("timestamp").reset_index(drop=True)
    split_idx = int(len(sorted_ratings) * (1 - test_ratio))
    train_df = sorted_ratings.iloc[:split_idx].reset_index(drop=True)
    test_df = sorted_ratings.iloc[split_idx:].reset_index(drop=True)
    return train_df, test_df


def dataset_overview(bundle: MovieLens100KBundle) -> dict[str, float | int]:
    ratings = bundle.ratings
    return {
        "n_ratings": int(len(ratings)),
        "n_users": int(ratings["user_id"].nunique()),
        "n_items": int(ratings["item_id"].nunique()),
        "rating_mean": float(ratings["rating"].mean()),
        "rating_median": float(ratings["rating"].median()),
        "sparsity": float(calculate_sparsity(ratings)),
    }


def build_relevance_sets(
    ratings: pd.DataFrame,
    min_rating: float = 4.0,
) -> dict[int, set[int]]:
    relevant = ratings.loc[ratings["rating"] >= min_rating, ["user_id", "item_id"]]
    grouped = relevant.groupby("user_id")["item_id"].apply(set)
    return grouped.to_dict()


def filter_seen_items(
    recommendations: dict[int, Iterable[int]],
    interactions: pd.DataFrame,
) -> dict[int, list[int]]:
    seen_items = interactions.groupby("user_id")["item_id"].apply(set).to_dict()
    filtered: dict[int, list[int]] = {}
    for user_id, items in recommendations.items():
        user_seen = seen_items.get(user_id, set())
        filtered[user_id] = [item for item in items if item not in user_seen]
    return filtered


def main() -> None:
    parser = argparse.ArgumentParser(description="MovieLens 100K dataset helper")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download and extract the official MovieLens 100K dataset.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the extracted dataset already exists.",
    )
    args = parser.parse_args()

    if args.download:
        dataset_path = download_movielens_100k(force=args.force)
        print(f"Dataset ready at: {dataset_path}")
        return

    dataset_path = ensure_movielens_100k(download_if_missing=False)
    print(f"Dataset found at: {dataset_path}")


if __name__ == "__main__":
    main()
