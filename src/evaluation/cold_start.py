from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from src.config import DEFAULT_TOP_K, HIGH_RATING_THRESHOLD, RANDOM_SEED
from src.evaluation.metrics import evaluate_recommendations
from src.features import build_tfidf_item_feature_matrix
from src.models import build_explicit_matrix, compute_similarity, popularity_scores


@dataclass(slots=True)
class NewUserRecommendationBundle:
    strategy: str
    recommendations: list[int]
    score_frame: pd.DataFrame


def score_new_user_content(
    observed_ratings: pd.DataFrame,
    item_feature_matrix: pd.DataFrame,
    *,
    center_rating_at: float = 3.0,
) -> pd.Series:
    """Score catalog items for a seed user using observed item metadata only."""
    available = observed_ratings.loc[
        observed_ratings["item_id"].isin(item_feature_matrix.index),
        ["item_id", "rating"],
    ].copy()
    if available.empty:
        return pd.Series(0.0, index=item_feature_matrix.index, dtype=float)

    weights = (available["rating"] - center_rating_at).clip(lower=0.1).to_numpy(dtype=float)
    vectors = item_feature_matrix.loc[available["item_id"]].to_numpy(dtype=float)
    profile = np.average(vectors, axis=0, weights=weights)

    profile_norm = np.linalg.norm(profile)
    if profile_norm == 0.0:
        return pd.Series(0.0, index=item_feature_matrix.index, dtype=float)

    catalog_vectors = item_feature_matrix.to_numpy(dtype=float)
    catalog_norms = np.linalg.norm(catalog_vectors, axis=1)
    catalog_norms[catalog_norms == 0.0] = 1.0
    scores = (catalog_vectors @ profile) / (catalog_norms * profile_norm)
    return pd.Series(scores, index=item_feature_matrix.index, dtype=float)


def score_new_user_item_cf(
    observed_ratings: pd.DataFrame,
    item_similarity: pd.DataFrame,
    *,
    center_rating_at: float = 3.0,
) -> pd.Series:
    """Score items from an item-item similarity matrix for a cold-start user."""
    available = observed_ratings.loc[
        observed_ratings["item_id"].isin(item_similarity.columns),
        ["item_id", "rating"],
    ].copy()
    if available.empty:
        return pd.Series(0.0, index=item_similarity.index, dtype=float)

    weights = (available["rating"] - center_rating_at).clip(lower=0.1).to_numpy(dtype=float)
    similarity_slice = item_similarity.loc[:, available["item_id"]].copy()
    weighted_scores = similarity_slice.mul(weights, axis=1).sum(axis=1) / float(np.abs(weights).sum())
    return weighted_scores.astype(float)


def blend_new_user_scores(
    cf_scores: pd.Series,
    content_scores: pd.Series,
    *,
    alpha: float = 0.85,
) -> pd.Series:
    if alpha < 0.0 or alpha > 1.0:
        raise ValueError("alpha must be between 0 and 1.")

    aligned_index = cf_scores.index.union(content_scores.index)
    cf_aligned = cf_scores.reindex(aligned_index).fillna(0.0)
    content_aligned = content_scores.reindex(aligned_index).fillna(0.0)

    def _minmax(series: pd.Series) -> pd.Series:
        minimum = float(series.min())
        maximum = float(series.max())
        if maximum == minimum:
            return pd.Series(0.0, index=series.index, dtype=float)
        return (series - minimum) / (maximum - minimum)

    cf_norm = _minmax(cf_aligned)
    content_norm = _minmax(content_aligned)
    return (alpha * cf_norm) + ((1.0 - alpha) * content_norm)


def recommend_for_new_user(
    observed_ratings: pd.DataFrame,
    *,
    popularity_ranking: Sequence[int],
    item_feature_matrix: pd.DataFrame,
    item_similarity: pd.DataFrame,
    top_k: int = DEFAULT_TOP_K,
    cf_switch_threshold: int = 5,
    hybrid_alpha: float = 0.85,
) -> NewUserRecommendationBundle:
    """Return recommendations for a new user as their observed history grows."""
    seen_items = set(observed_ratings["item_id"].tolist())

    if observed_ratings.empty:
        recommendations = [item_id for item_id in popularity_ranking if item_id not in seen_items][:top_k]
        score_frame = pd.DataFrame(
            {
                "item_id": recommendations,
                "score": list(reversed(range(1, len(recommendations) + 1))),
                "strategy": "popularity",
            }
        )
        return NewUserRecommendationBundle(
            strategy="popularity",
            recommendations=recommendations,
            score_frame=score_frame,
        )

    content_scores = score_new_user_content(observed_ratings, item_feature_matrix)
    cf_scores = score_new_user_item_cf(observed_ratings, item_similarity)

    if len(observed_ratings) < cf_switch_threshold:
        strategy = "content"
        final_scores = content_scores
    else:
        strategy = "hybrid"
        final_scores = blend_new_user_scores(cf_scores, content_scores, alpha=hybrid_alpha)

    ranked = (
        final_scores.loc[~final_scores.index.isin(seen_items)]
        .sort_values(ascending=False)
        .head(top_k)
        .rename("score")
        .reset_index()
        .rename(columns={"index": "item_id"})
    )
    ranked["strategy"] = strategy
    return NewUserRecommendationBundle(
        strategy=strategy,
        recommendations=ranked["item_id"].astype(int).tolist(),
        score_frame=ranked,
    )


def simulate_new_user_transition(
    ratings: pd.DataFrame,
    items: pd.DataFrame,
    *,
    thresholds: Iterable[int] = (0, 1, 3, 5, 10),
    top_k: int = DEFAULT_TOP_K,
    relevant_threshold: float = HIGH_RATING_THRESHOLD,
    cf_switch_threshold: int = 5,
    hybrid_alpha: float = 0.85,
    min_future_relevant: int = 1,
    max_users: int | None = 250,
    random_state: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate how recommendation quality changes as a new user provides ratings."""
    thresholds = sorted(set(int(value) for value in thresholds))
    if any(value < 0 for value in thresholds):
        raise ValueError("thresholds must be non-negative integers.")

    item_feature_matrix = build_tfidf_item_feature_matrix(items)
    item_similarity = compute_similarity(build_explicit_matrix(ratings), axis="item", metric="pearson").fillna(0.0)
    popularity_ranking = popularity_scores(ratings)["item_id"].astype(int).tolist()

    grouped_histories = {
        int(user_id): frame.sort_values("timestamp").reset_index(drop=True)
        for user_id, frame in ratings.groupby("user_id")
    }
    eligible_users = [
        user_id
        for user_id, history in grouped_histories.items()
        if len(history) >= (max(thresholds) + 5)
        and int((history["rating"] >= relevant_threshold).sum()) >= (min_future_relevant + 1)
    ]

    if max_users is not None and len(eligible_users) > max_users:
        rng = np.random.default_rng(random_state)
        eligible_users = sorted(rng.choice(eligible_users, size=max_users, replace=False).tolist())

    summary_rows: list[dict[str, float | int | str]] = []
    detail_rows: list[dict[str, float | int | str]] = []

    for threshold in thresholds:
        recommendations: dict[int, list[int]] = {}
        ground_truth: dict[int, set[int]] = {}

        for user_id in eligible_users:
            history = grouped_histories[user_id]
            observed = history.head(threshold).copy()
            future = history.iloc[threshold:].copy()
            relevant_items = set(future.loc[future["rating"] >= relevant_threshold, "item_id"].astype(int).tolist())
            if len(relevant_items) < min_future_relevant:
                continue

            bundle = recommend_for_new_user(
                observed_ratings=observed,
                popularity_ranking=popularity_ranking,
                item_feature_matrix=item_feature_matrix,
                item_similarity=item_similarity,
                top_k=top_k,
                cf_switch_threshold=cf_switch_threshold,
                hybrid_alpha=hybrid_alpha,
            )
            recommendations[user_id] = bundle.recommendations
            ground_truth[user_id] = relevant_items

        metrics, user_rows = evaluate_recommendations(
            recommendations,
            ground_truth,
            k=top_k,
            catalog=item_feature_matrix.index.tolist(),
            return_user_metrics=True,
        )
        stage = "popularity" if threshold == 0 else ("content" if threshold < cf_switch_threshold else "hybrid")
        summary_rows.append(
            {
                "observed_ratings": threshold,
                "stage": stage,
                **metrics,
            }
        )

        for row in user_rows:
            detail_rows.append(
                {
                    "observed_ratings": threshold,
                    "stage": stage,
                    **row,
                }
            )

    return pd.DataFrame(summary_rows), pd.DataFrame(detail_rows)
