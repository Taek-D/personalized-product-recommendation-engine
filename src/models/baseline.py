from __future__ import annotations

from typing import Iterable

import pandas as pd

from src.config import DEFAULT_TOP_K


DEFAULT_RATING_COLUMNS = ["user_id", "item_id", "rating"]


def validate_ratings_columns(ratings: pd.DataFrame) -> None:
    missing = [column for column in DEFAULT_RATING_COLUMNS if column not in ratings.columns]
    if missing:
        raise ValueError(f"Ratings DataFrame is missing required columns: {missing}")


def global_mean_rating(ratings: pd.DataFrame) -> float:
    validate_ratings_columns(ratings)
    return float(ratings["rating"].mean())


def user_mean_ratings(ratings: pd.DataFrame) -> pd.Series:
    validate_ratings_columns(ratings)
    return ratings.groupby("user_id")["rating"].mean().rename("user_mean_rating")


def item_mean_ratings(ratings: pd.DataFrame) -> pd.Series:
    validate_ratings_columns(ratings)
    return ratings.groupby("item_id")["rating"].mean().rename("item_mean_rating")


def popularity_scores(ratings: pd.DataFrame) -> pd.DataFrame:
    validate_ratings_columns(ratings)
    scores = (
        ratings.groupby("item_id")
        .agg(
            interaction_count=("user_id", "count"),
            mean_rating=("rating", "mean"),
        )
        .sort_values(["interaction_count", "mean_rating", "item_id"], ascending=[False, False, True])
        .reset_index()
    )
    return scores


def predict_global_mean(train_ratings: pd.DataFrame, test_ratings: pd.DataFrame) -> pd.DataFrame:
    validate_ratings_columns(train_ratings)
    validate_ratings_columns(test_ratings)
    baseline = global_mean_rating(train_ratings)
    predictions = test_ratings[["user_id", "item_id", "rating"]].copy()
    predictions["prediction"] = baseline
    predictions["baseline_model"] = "global_mean"
    return predictions


def predict_user_mean(train_ratings: pd.DataFrame, test_ratings: pd.DataFrame) -> pd.DataFrame:
    validate_ratings_columns(train_ratings)
    validate_ratings_columns(test_ratings)
    global_mean = global_mean_rating(train_ratings)
    user_means = user_mean_ratings(train_ratings)
    predictions = test_ratings[["user_id", "item_id", "rating"]].copy()
    predictions["prediction"] = predictions["user_id"].map(user_means).fillna(global_mean)
    predictions["baseline_model"] = "user_mean"
    return predictions


def recommend_popular_items(
    train_ratings: pd.DataFrame,
    user_ids: Iterable[int],
    top_k: int = DEFAULT_TOP_K,
    seen_interactions: pd.DataFrame | None = None,
) -> dict[int, list[int]]:
    validate_ratings_columns(train_ratings)
    ranking = popularity_scores(train_ratings)["item_id"].tolist()

    if seen_interactions is None:
        seen_interactions = train_ratings

    seen_by_user = seen_interactions.groupby("user_id")["item_id"].apply(set).to_dict()
    recommendations: dict[int, list[int]] = {}

    for user_id in user_ids:
        seen_items = seen_by_user.get(user_id, set())
        recommendations[user_id] = [item_id for item_id in ranking if item_id not in seen_items][:top_k]

    return recommendations


def baseline_summary(train_ratings: pd.DataFrame) -> dict[str, float | int]:
    scores = popularity_scores(train_ratings)
    return {
        "global_mean_rating": global_mean_rating(train_ratings),
        "n_users": int(train_ratings["user_id"].nunique()),
        "n_items": int(train_ratings["item_id"].nunique()),
        "most_popular_item_id": int(scores.iloc[0]["item_id"]) if not scores.empty else -1,
        "most_popular_item_interactions": int(scores.iloc[0]["interaction_count"]) if not scores.empty else 0,
    }
