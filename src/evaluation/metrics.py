from __future__ import annotations

from collections.abc import Collection, Hashable, Iterable, Mapping, Sequence
from math import log2, sqrt
from typing import Any


RecommendationList = Sequence[Hashable]
RecommendationDict = Mapping[Hashable, RecommendationList]
GroundTruthDict = Mapping[Hashable, Collection[Hashable]]


def _validate_k(k: int) -> int:
    if k <= 0:
        raise ValueError("k must be a positive integer.")
    return int(k)


def _unique_preserve_order(items: Iterable[Hashable]) -> list[Hashable]:
    seen: set[Hashable] = set()
    unique_items: list[Hashable] = []

    for item in items:
        if item in seen:
            continue
        seen.add(item)
        unique_items.append(item)

    return unique_items


def _top_k_items(recommended_items: RecommendationList, k: int) -> list[Hashable]:
    return _unique_preserve_order(recommended_items)[: _validate_k(k)]


def _relevant_items(relevant_items: Collection[Hashable]) -> set[Hashable]:
    return set(relevant_items)


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def precision_at_k(
    recommended_items: RecommendationList,
    relevant_items: Collection[Hashable],
    k: int,
) -> float:
    top_k = _top_k_items(recommended_items, k)
    if not top_k:
        return 0.0

    relevant = _relevant_items(relevant_items)
    hits = sum(item in relevant for item in top_k)
    return hits / len(top_k)


def recall_at_k(
    recommended_items: RecommendationList,
    relevant_items: Collection[Hashable],
    k: int,
) -> float:
    relevant = _relevant_items(relevant_items)
    if not relevant:
        return 0.0

    top_k = _top_k_items(recommended_items, k)
    hits = sum(item in relevant for item in top_k)
    return hits / len(relevant)


def ndcg_at_k(
    recommended_items: RecommendationList,
    relevant_items: Collection[Hashable],
    k: int,
) -> float:
    relevant = _relevant_items(relevant_items)
    if not relevant:
        return 0.0

    top_k = _top_k_items(recommended_items, k)
    if not top_k:
        return 0.0

    dcg = 0.0
    for rank, item in enumerate(top_k, start=1):
        if item in relevant:
            dcg += 1.0 / log2(rank + 1)

    ideal_hits = min(len(relevant), len(top_k))
    if ideal_hits == 0:
        return 0.0

    idcg = sum(1.0 / log2(rank + 1) for rank in range(1, ideal_hits + 1))
    if idcg == 0.0:
        return 0.0

    return dcg / idcg


def average_precision_at_k(
    recommended_items: RecommendationList,
    relevant_items: Collection[Hashable],
    k: int,
) -> float:
    relevant = _relevant_items(relevant_items)
    if not relevant:
        return 0.0

    top_k = _top_k_items(recommended_items, k)
    if not top_k:
        return 0.0

    hit_count = 0
    precision_sum = 0.0

    for rank, item in enumerate(top_k, start=1):
        if item not in relevant:
            continue
        hit_count += 1
        precision_sum += hit_count / rank

    if hit_count == 0:
        return 0.0

    normalizer = min(len(relevant), len(top_k))
    return precision_sum / normalizer


def map_at_k(
    recommendations: RecommendationDict,
    ground_truth: GroundTruthDict,
    k: int,
) -> float:
    scores = [
        average_precision_at_k(recommended_items, ground_truth[user_id], k)
        for user_id, recommended_items in recommendations.items()
        if user_id in ground_truth
    ]
    return _mean(scores)


def coverage(
    recommendations: RecommendationDict,
    catalog: Collection[Hashable],
) -> float:
    catalog_set = set(catalog)
    if not catalog_set:
        return 0.0

    recommended_items = {
        item
        for user_items in recommendations.values()
        for item in _unique_preserve_order(user_items)
        if item in catalog_set
    }
    return len(recommended_items) / len(catalog_set)


def _vectorize(values: Sequence[Any]) -> list[float]:
    return [float(value) for value in values]


def _coerce_item_feature_map(
    item_features: Any,
    item_ids: Sequence[Hashable] | None = None,
) -> dict[Hashable, list[float]]:
    if hasattr(item_features, "iterrows"):
        return {
            index: _vectorize(row.tolist())
            for index, row in item_features.iterrows()
        }

    if isinstance(item_features, Mapping):
        return {item_id: _vectorize(vector) for item_id, vector in item_features.items()}

    if item_ids is None:
        raise ValueError("item_ids must be provided when item_features is array-like.")

    rows = list(item_features)
    if len(item_ids) != len(rows):
        raise ValueError("item_ids length must match item_features row count.")

    return {
        item_id: _vectorize(vector)
        for item_id, vector in zip(item_ids, rows, strict=True)
    }


def _cosine_similarity(left_vector: Sequence[float], right_vector: Sequence[float]) -> float:
    if len(left_vector) != len(right_vector):
        raise ValueError("Feature vectors must share the same dimension.")

    numerator = sum(left * right for left, right in zip(left_vector, right_vector, strict=True))
    left_norm = sqrt(sum(value * value for value in left_vector))
    right_norm = sqrt(sum(value * value for value in right_vector))

    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0

    return numerator / (left_norm * right_norm)


def _lookup_similarity(
    item_similarity: Any,
    left_item: Hashable,
    right_item: Hashable,
) -> float | None:
    if (
        hasattr(item_similarity, "loc")
        and hasattr(item_similarity, "index")
        and hasattr(item_similarity, "columns")
    ):
        if left_item not in item_similarity.index or right_item not in item_similarity.columns:
            return None
        return float(item_similarity.loc[left_item, right_item])

    if isinstance(item_similarity, Mapping):
        left_row = item_similarity.get(left_item)
        if isinstance(left_row, Mapping) and right_item in left_row:
            return float(left_row[right_item])

        if (left_item, right_item) in item_similarity:
            return float(item_similarity[(left_item, right_item)])
        if (right_item, left_item) in item_similarity:
            return float(item_similarity[(right_item, left_item)])

    return None


def intra_list_diversity(
    recommended_items: RecommendationList,
    *,
    item_similarity: Any | None = None,
    item_features: Any | None = None,
    item_ids: Sequence[Hashable] | None = None,
    k: int | None = None,
) -> float:
    items = _unique_preserve_order(recommended_items)
    if k is not None:
        items = items[: _validate_k(k)]

    if len(items) < 2:
        return 0.0

    feature_map: dict[Hashable, list[float]] | None = None
    if item_similarity is None:
        if item_features is None:
            raise ValueError(
                "Provide either item_similarity or item_features for diversity evaluation."
            )
        feature_map = _coerce_item_feature_map(item_features, item_ids=item_ids)

    dissimilarities: list[float] = []
    for index, left_item in enumerate(items[:-1]):
        for right_item in items[index + 1 :]:
            similarity: float | None
            if item_similarity is not None:
                similarity = _lookup_similarity(item_similarity, left_item, right_item)
            else:
                if left_item not in feature_map or right_item not in feature_map:
                    continue
                similarity = _cosine_similarity(feature_map[left_item], feature_map[right_item])

            if similarity is None:
                continue

            dissimilarities.append(1.0 - similarity)

    return _mean(dissimilarities)


def evaluate_recommendations(
    recommendations: RecommendationDict,
    ground_truth: GroundTruthDict,
    *,
    k: int = 10,
    catalog: Collection[Hashable] | None = None,
    item_similarity: Any | None = None,
    item_features: Any | None = None,
    item_ids: Sequence[Hashable] | None = None,
    return_user_metrics: bool = False,
) -> dict[str, float] | tuple[dict[str, float], list[dict[str, Any]]]:
    _validate_k(k)

    user_rows: list[dict[str, Any]] = []
    for user_id, recommended_items in recommendations.items():
        if user_id not in ground_truth:
            continue

        row = {
            "user_id": user_id,
            f"precision@{k}": precision_at_k(
                recommended_items, ground_truth[user_id], k
            ),
            f"recall@{k}": recall_at_k(recommended_items, ground_truth[user_id], k),
            f"ndcg@{k}": ndcg_at_k(recommended_items, ground_truth[user_id], k),
            f"ap@{k}": average_precision_at_k(
                recommended_items, ground_truth[user_id], k
            ),
        }

        if item_similarity is not None or item_features is not None:
            row["intra_list_diversity"] = intra_list_diversity(
                recommended_items,
                item_similarity=item_similarity,
                item_features=item_features,
                item_ids=item_ids,
                k=k,
            )

        user_rows.append(row)

    summary: dict[str, float] = {
        "users_evaluated": float(len(user_rows)),
        f"precision@{k}": _mean([row[f"precision@{k}"] for row in user_rows]),
        f"recall@{k}": _mean([row[f"recall@{k}"] for row in user_rows]),
        f"ndcg@{k}": _mean([row[f"ndcg@{k}"] for row in user_rows]),
        f"map@{k}": _mean([row[f"ap@{k}"] for row in user_rows]),
    }

    if any("intra_list_diversity" in row for row in user_rows):
        summary["intra_list_diversity"] = _mean(
            [
                float(row["intra_list_diversity"])
                for row in user_rows
                if "intra_list_diversity" in row
            ]
        )

    if catalog is not None:
        summary["coverage"] = coverage(recommendations, catalog)

    if return_user_metrics:
        return summary, user_rows

    return summary
