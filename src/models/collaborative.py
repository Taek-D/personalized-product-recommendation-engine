from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
import pandas as pd

from src.config import DEFAULT_TOP_K, HIGH_RATING_THRESHOLD, RATING_SCALE
from src.data import create_implicit_feedback, create_user_item_matrix

SimilarityMetric = Literal["cosine", "pearson", "jaccard"]
AxisType = Literal["user", "item"]


@dataclass(slots=True)
class SurpriseModelBundle:
    algo_name: str
    algo: object
    trainset: object


@dataclass(slots=True)
class ImplicitALSBundle:
    model: object
    user_item_matrix: object
    user_index: dict[int, int]
    item_index: dict[int, int]
    reverse_user_index: dict[int, int]
    reverse_item_index: dict[int, int]


def build_explicit_matrix(ratings: pd.DataFrame, fill_value: float = 0.0) -> pd.DataFrame:
    return create_user_item_matrix(ratings, value_column="rating", fill_value=fill_value)


def build_implicit_matrix(
    ratings: pd.DataFrame,
    threshold: float = HIGH_RATING_THRESHOLD,
    fill_value: float = 0.0,
) -> pd.DataFrame:
    implicit_feedback = create_implicit_feedback(ratings, threshold=threshold)
    implicit_feedback = implicit_feedback.rename(columns={"interaction": "implicit_signal"})
    return create_user_item_matrix(
        implicit_feedback,
        value_column="implicit_signal",
        fill_value=fill_value,
    )


def compute_similarity(
    matrix: pd.DataFrame,
    axis: AxisType = "user",
    metric: SimilarityMetric = "cosine",
) -> pd.DataFrame:
    base = matrix if axis == "user" else matrix.T
    values = base.to_numpy(dtype=float)
    labels = base.index

    if metric == "cosine":
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        normalized = values / norms
        similarity = normalized @ normalized.T
    elif metric == "pearson":
        similarity = np.corrcoef(values)
        similarity = np.nan_to_num(similarity, nan=0.0)
    elif metric == "jaccard":
        binary = (values > 0).astype(int)
        intersections = binary @ binary.T
        row_sums = binary.sum(axis=1, keepdims=True)
        unions = row_sums + row_sums.T - intersections
        unions[unions == 0] = 1
        similarity = intersections / unions
    else:
        raise ValueError(f"Unsupported similarity metric: {metric}")

    np.fill_diagonal(similarity, 1.0)
    return pd.DataFrame(similarity, index=labels, columns=labels)


def _top_k_neighbors(similarity: pd.Series, k: int) -> pd.Series:
    return similarity.drop(labels=[similarity.name], errors="ignore").sort_values(ascending=False).head(k)


def predict_user_based_scores(
    ratings: pd.DataFrame,
    k: int = 20,
    metric: SimilarityMetric = "cosine",
    fill_value: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    user_item = build_explicit_matrix(ratings, fill_value=fill_value)
    user_similarity = compute_similarity(user_item, axis="user", metric=metric)
    user_means = user_item.replace(0, np.nan).mean(axis=1).fillna(0.0)
    centered = user_item.sub(user_means, axis=0).fillna(0.0)

    predictions = pd.DataFrame(index=user_item.index, columns=user_item.columns, dtype=float)
    for user_id in user_item.index:
        neighbors = _top_k_neighbors(user_similarity.loc[user_id].rename(user_id), k=k)
        weights = neighbors[neighbors > 0]
        if weights.empty:
            predictions.loc[user_id] = user_means.loc[user_id]
            continue
        neighbor_ratings = centered.loc[weights.index]
        numerator = (neighbor_ratings.T * weights).T.sum(axis=0)
        denominator = weights.abs().sum()
        predictions.loc[user_id] = user_means.loc[user_id] + numerator / denominator

    return predictions, user_similarity


def predict_item_based_scores(
    ratings: pd.DataFrame,
    k: int = 20,
    metric: SimilarityMetric = "cosine",
    fill_value: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    user_item = build_explicit_matrix(ratings, fill_value=fill_value)
    item_similarity = compute_similarity(user_item, axis="item", metric=metric)

    predictions = pd.DataFrame(index=user_item.index, columns=user_item.columns, dtype=float)
    for item_id in user_item.columns:
        neighbors = _top_k_neighbors(item_similarity.loc[item_id].rename(item_id), k=k)
        weights = neighbors[neighbors > 0]
        if weights.empty:
            predictions[item_id] = 0.0
            continue
        neighbor_ratings = user_item[weights.index]
        numerator = (neighbor_ratings * weights).sum(axis=1)
        denominator = weights.abs().sum()
        predictions[item_id] = numerator / denominator

    return predictions, item_similarity


def recommend_from_score_matrix(
    score_matrix: pd.DataFrame,
    interactions: pd.DataFrame,
    user_ids: Iterable[int] | None = None,
    top_k: int = DEFAULT_TOP_K,
) -> dict[int, list[int]]:
    seen_items = interactions.groupby("user_id")["item_id"].apply(set).to_dict()
    target_users = list(user_ids) if user_ids is not None else list(score_matrix.index)
    recommendations: dict[int, list[int]] = {}

    for user_id in target_users:
        scores = score_matrix.loc[user_id].sort_values(ascending=False)
        filtered_items = [item_id for item_id in scores.index if item_id not in seen_items.get(user_id, set())]
        recommendations[user_id] = filtered_items[:top_k]

    return recommendations


def predict_from_score_matrix(
    score_matrix: pd.DataFrame,
    test_ratings: pd.DataFrame,
    default_prediction: float | None = None,
    clip_range: tuple[float, float] | None = RATING_SCALE,
    model_name: str = "memory_cf",
) -> pd.DataFrame:
    predictions = test_ratings[["user_id", "item_id", "rating"]].copy()

    if default_prediction is None:
        default_prediction = float(np.nanmean(score_matrix.to_numpy(dtype=float)))

    def lookup_prediction(row: pd.Series) -> float:
        user_id = row["user_id"]
        item_id = row["item_id"]
        if user_id in score_matrix.index and item_id in score_matrix.columns:
            value = score_matrix.at[user_id, item_id]
            if pd.notna(value):
                prediction = float(value)
                if clip_range is not None:
                    prediction = float(np.clip(prediction, clip_range[0], clip_range[1]))
                return prediction
        return float(default_prediction)

    predictions["prediction"] = predictions.apply(lookup_prediction, axis=1)
    predictions["baseline_model"] = model_name
    return predictions


def predict_surprise_frame(
    bundle: SurpriseModelBundle,
    test_ratings: pd.DataFrame,
) -> pd.DataFrame:
    predictions = []
    for row in test_ratings[["user_id", "item_id", "rating"]].itertuples(index=False):
        estimate = bundle.algo.predict(row.user_id, row.item_id, r_ui=row.rating).est
        predictions.append(
            {
                "user_id": row.user_id,
                "item_id": row.item_id,
                "rating": row.rating,
                "prediction": estimate,
                "baseline_model": bundle.algo_name,
            }
        )
    return pd.DataFrame(predictions)


def recommend_surprise_top_k(
    bundle: SurpriseModelBundle,
    train_ratings: pd.DataFrame,
    user_ids: Iterable[int] | None = None,
    item_ids: Iterable[int] | None = None,
    top_k: int = DEFAULT_TOP_K,
) -> dict[int, list[int]]:
    seen_items = train_ratings.groupby("user_id")["item_id"].apply(set).to_dict()
    candidate_items = list(item_ids) if item_ids is not None else sorted(train_ratings["item_id"].unique())
    target_users = list(user_ids) if user_ids is not None else sorted(train_ratings["user_id"].unique())
    recommendations: dict[int, list[int]] = {}

    for user_id in target_users:
        user_seen = seen_items.get(user_id, set())
        scores = []
        for item_id in candidate_items:
            if item_id in user_seen:
                continue
            estimate = bundle.algo.predict(user_id, item_id).est
            scores.append((item_id, estimate))
        scores.sort(key=lambda pair: pair[1], reverse=True)
        recommendations[user_id] = [item_id for item_id, _ in scores[:top_k]]

    return recommendations


def fit_surprise_model(
    train_ratings: pd.DataFrame,
    algo_name: Literal["svd", "svdpp", "nmf"] = "svd",
    rating_scale: tuple[int, int] = RATING_SCALE,
    **kwargs,
) -> SurpriseModelBundle:
    try:
        from surprise import Dataset, NMF, Reader, SVD, SVDpp
    except ImportError as error:  # pragma: no cover - depends on optional packages
        raise ImportError(
            "surprise is not installed. Install dependencies from requirements.txt first."
        ) from error

    algo_lookup = {
        "svd": SVD,
        "svdpp": SVDpp,
        "nmf": NMF,
    }
    if algo_name not in algo_lookup:
        raise ValueError(f"Unsupported Surprise algorithm: {algo_name}")

    reader = Reader(rating_scale=rating_scale)
    dataset = Dataset.load_from_df(train_ratings[["user_id", "item_id", "rating"]], reader)
    trainset = dataset.build_full_trainset()
    algo = algo_lookup[algo_name](**kwargs)
    algo.fit(trainset)
    return SurpriseModelBundle(algo_name=algo_name, algo=algo, trainset=trainset)


def fit_implicit_als(
    ratings: pd.DataFrame,
    factors: int = 50,
    regularization: float = 0.01,
    iterations: int = 20,
    alpha: float = 40.0,
) -> ImplicitALSBundle:
    try:
        import scipy.sparse as sp
        from implicit.als import AlternatingLeastSquares
    except ImportError as error:  # pragma: no cover - depends on optional packages
        raise ImportError(
            "implicit and scipy are required for ALS. Install dependencies from requirements.txt first."
        ) from error

    implicit_matrix = build_implicit_matrix(ratings)
    weighted = (implicit_matrix * alpha).astype(float)

    user_ids = list(weighted.index)
    item_ids = list(weighted.columns)
    user_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
    item_index = {item_id: idx for idx, item_id in enumerate(item_ids)}
    reverse_user_index = {idx: user_id for user_id, idx in user_index.items()}
    reverse_item_index = {idx: item_id for item_id, idx in item_index.items()}

    sparse_matrix = sp.csr_matrix(weighted.to_numpy())
    model = AlternatingLeastSquares(
        factors=factors,
        regularization=regularization,
        iterations=iterations,
    )
    model.fit(sparse_matrix)
    return ImplicitALSBundle(
        model=model,
        user_item_matrix=sparse_matrix,
        user_index=user_index,
        item_index=item_index,
        reverse_user_index=reverse_user_index,
        reverse_item_index=reverse_item_index,
    )


def recommend_als(
    bundle: ImplicitALSBundle,
    user_ids: Iterable[int],
    top_k: int = DEFAULT_TOP_K,
    filter_already_liked_items: bool = True,
) -> dict[int, list[int]]:
    recommendations: dict[int, list[int]] = {}

    for user_id in user_ids:
        if user_id not in bundle.user_index:
            recommendations[user_id] = []
            continue

        inner_user_id = bundle.user_index[user_id]
        item_ids, _ = bundle.model.recommend(
            inner_user_id,
            bundle.user_item_matrix[inner_user_id],
            N=top_k,
            filter_already_liked_items=filter_already_liked_items,
        )
        recommendations[user_id] = [bundle.reverse_item_index[int(item_id)] for item_id in item_ids]

    return recommendations
