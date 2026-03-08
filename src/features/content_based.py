from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from src.config import DEFAULT_TOP_K, HIGH_RATING_THRESHOLD

GENRE_COLUMNS = [
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


def build_item_feature_matrix(
    items: pd.DataFrame,
    include_genres: bool = True,
    include_release_decade: bool = True,
    normalize_vectors: bool = True,
) -> pd.DataFrame:
    features: list[pd.DataFrame] = []

    if include_genres:
        genre_features = items[["item_id", *GENRE_COLUMNS]].copy().set_index("item_id")
        features.append(genre_features.astype(float))

    if include_release_decade:
        decade_frame = (
            items[["item_id", "release_decade"]]
            .copy()
            .assign(release_decade=lambda df: df["release_decade"].fillna(-1).astype(int).astype(str))
        )
        decade_dummies = pd.get_dummies(decade_frame.set_index("item_id")["release_decade"], prefix="decade")
        features.append(decade_dummies.astype(float))

    if not features:
        raise ValueError("At least one feature family must be enabled.")

    feature_matrix = pd.concat(features, axis=1).fillna(0.0)
    feature_matrix = feature_matrix.loc[~feature_matrix.index.duplicated(keep="first")]

    if normalize_vectors:
        normalized = normalize(feature_matrix.to_numpy(dtype=float), norm="l2")
        feature_matrix = pd.DataFrame(normalized, index=feature_matrix.index, columns=feature_matrix.columns)

    return feature_matrix


def _clean_title_for_text_features(title: str) -> str:
    normalized = re.sub(r"\((?:19|20)\d{2}\)", " ", str(title).lower())
    normalized = re.sub(r"[^a-z0-9 ]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def build_tfidf_item_feature_matrix(
    items: pd.DataFrame,
    *,
    max_features: int = 1500,
    min_df: int = 2,
    ngram_range: tuple[int, int] = (1, 2),
    stop_words: str | None = "english",
) -> pd.DataFrame:
    """Build a TF-IDF item feature matrix from MovieLens metadata.

    MovieLens 100K does not provide free-form tag tables, so we approximate a
    text feature view with title tokens + genre tokens + release decade tokens.
    """

    metadata = items[["item_id", "title", "genres", "release_decade"]].copy()
    metadata["metadata_text"] = metadata.apply(
        lambda row: " ".join(
            filter(
                None,
                [
                    _clean_title_for_text_features(row["title"]),
                    " ".join(row["genres"]) if isinstance(row["genres"], list) else "",
                    (
                        f"decade_{int(row['release_decade'])}"
                        if pd.notna(row["release_decade"])
                        else "decade_unknown"
                    ),
                ],
            )
        ),
        axis=1,
    )

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        ngram_range=ngram_range,
        stop_words=stop_words,
    )
    tfidf = vectorizer.fit_transform(metadata["metadata_text"])
    return pd.DataFrame(
        tfidf.toarray(),
        index=metadata["item_id"],
        columns=[f"tfidf_{term}" for term in vectorizer.get_feature_names_out()],
        dtype=float,
    )


def build_user_profile_matrix(
    ratings: pd.DataFrame,
    item_feature_matrix: pd.DataFrame,
    min_rating: float = HIGH_RATING_THRESHOLD,
    rating_weight_power: float = 1.0,
    center_rating_at: float = 3.0,
    normalize_vectors: bool = True,
) -> pd.DataFrame:
    positive_ratings = ratings.loc[ratings["rating"] >= min_rating, ["user_id", "item_id", "rating"]].copy()
    positive_ratings = positive_ratings[positive_ratings["item_id"].isin(item_feature_matrix.index)]
    if positive_ratings.empty:
        return pd.DataFrame(columns=item_feature_matrix.columns, dtype=float)

    positive_ratings["weight"] = (positive_ratings["rating"] - center_rating_at).clip(lower=0.1) ** rating_weight_power

    records: list[tuple[int, np.ndarray]] = []
    for user_id, user_rows in positive_ratings.groupby("user_id"):
        item_ids = user_rows["item_id"].tolist()
        weights = user_rows["weight"].to_numpy(dtype=float)
        item_vectors = item_feature_matrix.loc[item_ids].to_numpy(dtype=float)
        profile = np.average(item_vectors, axis=0, weights=weights)
        records.append((int(user_id), profile))

    user_profiles = pd.DataFrame(
        [profile for _, profile in records],
        index=[user_id for user_id, _ in records],
        columns=item_feature_matrix.columns,
        dtype=float,
    )

    if normalize_vectors and not user_profiles.empty:
        normalized = normalize(user_profiles.to_numpy(dtype=float), norm="l2")
        user_profiles = pd.DataFrame(normalized, index=user_profiles.index, columns=user_profiles.columns)

    return user_profiles


def score_user_item_content(
    user_profile_matrix: pd.DataFrame,
    item_feature_matrix: pd.DataFrame,
) -> pd.DataFrame:
    if user_profile_matrix.empty or item_feature_matrix.empty:
        return pd.DataFrame(index=user_profile_matrix.index, columns=item_feature_matrix.index, dtype=float)

    scores = cosine_similarity(
        user_profile_matrix.to_numpy(dtype=float),
        item_feature_matrix.to_numpy(dtype=float),
    )
    return pd.DataFrame(scores, index=user_profile_matrix.index, columns=item_feature_matrix.index)


def build_item_similarity_matrix(item_feature_matrix: pd.DataFrame) -> pd.DataFrame:
    if item_feature_matrix.empty:
        return pd.DataFrame(index=item_feature_matrix.index, columns=item_feature_matrix.index, dtype=float)
    similarity = cosine_similarity(item_feature_matrix.to_numpy(dtype=float))
    return pd.DataFrame(similarity, index=item_feature_matrix.index, columns=item_feature_matrix.index)


def recommend_content_based(
    ratings: pd.DataFrame,
    item_feature_matrix: pd.DataFrame,
    user_ids: Iterable[int] | None = None,
    top_k: int = DEFAULT_TOP_K,
    min_rating: float = HIGH_RATING_THRESHOLD,
) -> tuple[dict[int, list[int]], pd.DataFrame, pd.DataFrame]:
    user_profiles = build_user_profile_matrix(ratings, item_feature_matrix, min_rating=min_rating)
    score_matrix = score_user_item_content(user_profiles, item_feature_matrix)
    seen_items = ratings.groupby("user_id")["item_id"].apply(set).to_dict()

    target_users = list(user_ids) if user_ids is not None else list(score_matrix.index)
    recommendations: dict[int, list[int]] = {}

    for user_id in target_users:
        if user_id not in score_matrix.index:
            recommendations[user_id] = []
            continue
        user_scores = score_matrix.loc[user_id].sort_values(ascending=False)
        filtered_items = [item_id for item_id in user_scores.index if item_id not in seen_items.get(user_id, set())]
        recommendations[user_id] = filtered_items[:top_k]

    return recommendations, user_profiles, score_matrix


def explain_user_profile(
    user_id: int,
    ratings: pd.DataFrame,
    items: pd.DataFrame,
    min_rating: float = HIGH_RATING_THRESHOLD,
    top_n: int = 5,
) -> pd.DataFrame:
    merged = ratings.merge(items[["item_id", "title", "genres", "release_year"]], on="item_id", how="left")
    positives = merged[(merged["user_id"] == user_id) & (merged["rating"] >= min_rating)]
    return positives.sort_values(["rating", "timestamp"], ascending=[False, False]).head(top_n)
