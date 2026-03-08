from __future__ import annotations

from typing import Iterable, Literal

import numpy as np
import pandas as pd

from src.config import COLD_START_MIN_INTERACTIONS

NormalizationStrategy = Literal["none", "user_minmax", "user_zscore"]


def align_score_matrices(
    *score_matrices: pd.DataFrame,
    fill_value: float = 0.0,
) -> list[pd.DataFrame]:
    if not score_matrices:
        return []

    all_users = sorted({user_id for matrix in score_matrices for user_id in matrix.index})
    all_items = sorted({item_id for matrix in score_matrices for item_id in matrix.columns})
    return [
        matrix.reindex(index=all_users, columns=all_items).fillna(fill_value)
        for matrix in score_matrices
    ]


def normalize_score_matrix(
    score_matrix: pd.DataFrame,
    strategy: NormalizationStrategy = "user_minmax",
    fill_value: float = 0.0,
) -> pd.DataFrame:
    matrix = score_matrix.fillna(fill_value)
    if strategy == "none":
        return matrix

    values = matrix.to_numpy(dtype=float)

    if strategy == "user_minmax":
        row_min = values.min(axis=1, keepdims=True)
        row_max = values.max(axis=1, keepdims=True)
        denominator = np.where((row_max - row_min) == 0.0, 1.0, row_max - row_min)
        normalized = (values - row_min) / denominator
    elif strategy == "user_zscore":
        row_mean = values.mean(axis=1, keepdims=True)
        row_std = values.std(axis=1, keepdims=True)
        denominator = np.where(row_std == 0.0, 1.0, row_std)
        normalized = (values - row_mean) / denominator
    else:
        raise ValueError(f"Unsupported normalization strategy: {strategy}")

    return pd.DataFrame(normalized, index=matrix.index, columns=matrix.columns, dtype=float)


def blend_score_matrices(
    primary_score_matrix: pd.DataFrame,
    secondary_score_matrix: pd.DataFrame,
    *,
    alpha: float = 0.85,
    beta: float | None = None,
    normalization: NormalizationStrategy = "user_minmax",
    fill_value: float = 0.0,
) -> pd.DataFrame:
    if beta is None:
        beta = 1.0 - alpha
    if alpha < 0.0 or beta < 0.0:
        raise ValueError("alpha and beta must be non-negative.")
    if alpha == 0.0 and beta == 0.0:
        raise ValueError("alpha and beta cannot both be zero.")

    primary_aligned, secondary_aligned = align_score_matrices(
        primary_score_matrix,
        secondary_score_matrix,
        fill_value=fill_value,
    )
    primary_normalized = normalize_score_matrix(
        primary_aligned,
        strategy=normalization,
        fill_value=fill_value,
    )
    secondary_normalized = normalize_score_matrix(
        secondary_aligned,
        strategy=normalization,
        fill_value=fill_value,
    )

    denominator = alpha + beta
    blended = (alpha * primary_normalized) + (beta * secondary_normalized)
    return blended / denominator


def switch_score_matrices(
    primary_score_matrix: pd.DataFrame,
    secondary_score_matrix: pd.DataFrame,
    secondary_user_ids: Iterable[int],
    *,
    fill_value: float = 0.0,
) -> pd.DataFrame:
    all_users = sorted(set(primary_score_matrix.index) | set(secondary_score_matrix.index))
    all_items = sorted(set(primary_score_matrix.columns) | set(secondary_score_matrix.columns))

    primary_aligned = primary_score_matrix.reindex(index=all_users, columns=all_items)
    secondary_aligned = secondary_score_matrix.reindex(index=all_users, columns=all_items)

    def _fill_for_ranking(matrix: pd.DataFrame) -> pd.DataFrame:
        filled = matrix.copy()
        row_min = filled.min(axis=1, skipna=True).fillna(fill_value)
        for user_id in filled.index:
            floor_value = float(row_min.loc[user_id]) - 1.0
            filled.loc[user_id] = filled.loc[user_id].fillna(floor_value)
        return filled

    switched = _fill_for_ranking(primary_aligned)
    secondary_rankable = _fill_for_ranking(secondary_aligned)

    for user_id in secondary_user_ids:
        if user_id in secondary_rankable.index:
            switched.loc[user_id] = secondary_rankable.loc[user_id]

    return switched


def switch_recommendations(
    primary_recommendations: dict[int, list[int]],
    secondary_recommendations: dict[int, list[int]],
    secondary_user_ids: Iterable[int],
) -> dict[int, list[int]]:
    all_users = set(primary_recommendations) | set(secondary_recommendations)
    switched = {
        user_id: list(primary_recommendations.get(user_id, secondary_recommendations.get(user_id, [])))
        for user_id in all_users
    }

    for user_id in secondary_user_ids:
        if user_id in secondary_recommendations:
            switched[user_id] = list(secondary_recommendations[user_id])

    return switched


def users_below_interaction_threshold(
    interactions: pd.DataFrame,
    threshold: int = COLD_START_MIN_INTERACTIONS,
) -> pd.Index:
    if threshold <= 0:
        raise ValueError("threshold must be a positive integer.")

    user_counts = interactions.groupby("user_id")["item_id"].count()
    return user_counts[user_counts < threshold].index


def user_activity_segments(
    interactions: pd.DataFrame,
    *,
    cold_threshold: int = COLD_START_MIN_INTERACTIONS,
    warm_threshold: int = 20,
) -> pd.Series:
    if cold_threshold <= 0 or warm_threshold <= cold_threshold:
        raise ValueError("warm_threshold must be greater than cold_threshold, both positive.")

    user_counts = interactions.groupby("user_id")["item_id"].count()
    segments = pd.Series(index=user_counts.index, dtype="object")
    segments.loc[user_counts < cold_threshold] = "cold_user"
    segments.loc[(user_counts >= cold_threshold) & (user_counts < warm_threshold)] = "warm_user"
    segments.loc[user_counts >= warm_threshold] = "power_user"
    return segments.rename("activity_segment")


def recommendation_cold_item_share(
    recommendations: dict[int, list[int]],
    cold_item_ids: Iterable[int],
) -> float:
    cold_items = set(cold_item_ids)
    if not cold_items or not recommendations:
        return 0.0

    shares: list[float] = []
    for items in recommendations.values():
        if not items:
            shares.append(0.0)
            continue
        shares.append(sum(item_id in cold_items for item_id in items) / len(items))

    return float(np.mean(shares)) if shares else 0.0
