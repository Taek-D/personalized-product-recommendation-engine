from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import DEFAULT_TOP_K, HIGH_RATING_THRESHOLD, METRICS_DIR, RANDOM_SEED
from src.data import build_relevance_sets, filter_seen_items, load_bundle, random_train_test_split
from src.evaluation import evaluate_recommendations
from src.models import (
    fit_implicit_als,
    fit_surprise_model,
    predict_global_mean,
    predict_from_score_matrix,
    predict_item_based_scores,
    predict_surprise_frame,
    predict_user_based_scores,
    predict_user_mean,
    recommend_als,
    recommend_from_score_matrix,
    recommend_popular_items,
    recommend_surprise_top_k,
)


def rmse_from_frame(predictions: pd.DataFrame) -> float:
    error = predictions["rating"] - predictions["prediction"]
    return float(np.sqrt(np.mean(np.square(error))))


def evaluate_topk_model(
    model_name: str,
    recommendations: dict[int, list[int]],
    *,
    train_df: pd.DataFrame,
    relevant_items: dict[int, set[int]],
    catalog: list[int],
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, float | str]:
    filtered = filter_seen_items(recommendations, train_df)
    summary = evaluate_recommendations(
        recommendations=filtered,
        ground_truth=relevant_items,
        k=top_k,
        catalog=catalog,
    )
    return {"model_name": model_name, **summary}


def main() -> None:
    bundle = load_bundle(download_if_missing=False)
    train_df, test_df = random_train_test_split(bundle.ratings, test_size=0.2, random_state=RANDOM_SEED)
    relevant_items = build_relevance_sets(test_df, min_rating=HIGH_RATING_THRESHOLD)
    catalog = sorted(train_df["item_id"].unique())
    target_user_ids = sorted(test_df["user_id"].unique())
    global_fallback = float(train_df["rating"].mean())

    rows: list[dict[str, float | int | str]] = []

    global_mean_predictions = predict_global_mean(train_df, test_df)
    user_mean_predictions = predict_user_mean(train_df, test_df)
    popularity_recommendations = recommend_popular_items(
        train_df,
        user_ids=target_user_ids,
        top_k=DEFAULT_TOP_K,
    )

    rows.append(
        {
            "model_name": "global_mean",
            "family": "baseline",
            "rmse": rmse_from_frame(global_mean_predictions),
        }
    )
    rows.append(
        {
            "model_name": "user_mean",
            "family": "baseline",
            "rmse": rmse_from_frame(user_mean_predictions),
        }
    )
    popularity_row = evaluate_topk_model(
        "popularity_baseline",
        popularity_recommendations,
        train_df=train_df,
        relevant_items=relevant_items,
        catalog=catalog,
    )
    popularity_row["family"] = "baseline"
    rows.append(popularity_row)

    for similarity_name in ["cosine", "pearson", "jaccard"]:
        for n_neighbors in [10, 20, 40, 80]:
            user_score_matrix, _ = predict_user_based_scores(train_df, k=n_neighbors, metric=similarity_name)
            item_score_matrix, _ = predict_item_based_scores(train_df, k=n_neighbors, metric=similarity_name)

            user_prediction_df = predict_from_score_matrix(
                user_score_matrix,
                test_df,
                default_prediction=global_fallback,
                model_name=f"user_cf_{similarity_name}_k{n_neighbors}",
            )
            item_prediction_df = predict_from_score_matrix(
                item_score_matrix,
                test_df,
                default_prediction=global_fallback,
                model_name=f"item_cf_{similarity_name}_k{n_neighbors}",
            )

            user_recommendations = recommend_from_score_matrix(
                user_score_matrix,
                train_df,
                user_ids=target_user_ids,
                top_k=DEFAULT_TOP_K,
            )
            item_recommendations = recommend_from_score_matrix(
                item_score_matrix,
                train_df,
                user_ids=target_user_ids,
                top_k=DEFAULT_TOP_K,
            )

            user_row = evaluate_topk_model(
                f"user_cf_{similarity_name}_k{n_neighbors}",
                user_recommendations,
                train_df=train_df,
                relevant_items=relevant_items,
                catalog=catalog,
            )
            user_row["rmse"] = rmse_from_frame(user_prediction_df)
            user_row["family"] = "memory_cf"
            rows.append(user_row)

            item_row = evaluate_topk_model(
                f"item_cf_{similarity_name}_k{n_neighbors}",
                item_recommendations,
                train_df=train_df,
                relevant_items=relevant_items,
                catalog=catalog,
            )
            item_row["rmse"] = rmse_from_frame(item_prediction_df)
            item_row["family"] = "memory_cf"
            rows.append(item_row)

    surprise_configs = [
        ("svd", {"n_factors": 20, "reg_all": 0.02, "random_state": RANDOM_SEED}),
        ("svd", {"n_factors": 50, "reg_all": 0.02, "random_state": RANDOM_SEED}),
        ("svd", {"n_factors": 100, "reg_all": 0.05, "random_state": RANDOM_SEED}),
        ("svdpp", {"n_factors": 20, "reg_all": 0.02, "random_state": RANDOM_SEED}),
        ("svdpp", {"n_factors": 50, "reg_all": 0.05, "random_state": RANDOM_SEED}),
        ("nmf", {"n_factors": 20, "random_state": RANDOM_SEED}),
        ("nmf", {"n_factors": 50, "random_state": RANDOM_SEED}),
        ("nmf", {"n_factors": 100, "random_state": RANDOM_SEED}),
    ]

    for algo_name, kwargs in surprise_configs:
        bundle_model = fit_surprise_model(train_df, algo_name=algo_name, **kwargs)
        prediction_df = predict_surprise_frame(bundle_model, test_df)
        recommendations = recommend_surprise_top_k(
            bundle_model,
            train_df,
            user_ids=target_user_ids,
            top_k=DEFAULT_TOP_K,
        )
        suffix = "_".join(f"{key}{value}" for key, value in kwargs.items() if key != "random_state")
        model_name = f"{algo_name}_{suffix}"
        row = evaluate_topk_model(
            model_name,
            recommendations,
            train_df=train_df,
            relevant_items=relevant_items,
            catalog=catalog,
        )
        row["rmse"] = rmse_from_frame(prediction_df)
        row["family"] = "model_cf"
        rows.append(row)

    als_configs = [
        {"factors": 32, "regularization": 0.01, "iterations": 15, "alpha": 20.0},
        {"factors": 32, "regularization": 0.05, "iterations": 20, "alpha": 40.0},
        {"factors": 64, "regularization": 0.01, "iterations": 15, "alpha": 20.0},
        {"factors": 64, "regularization": 0.05, "iterations": 20, "alpha": 40.0},
    ]

    for kwargs in als_configs:
        bundle_model = fit_implicit_als(train_df, **kwargs)
        recommendations = recommend_als(
            bundle_model,
            user_ids=target_user_ids,
            top_k=DEFAULT_TOP_K,
        )
        suffix = "_".join(f"{key}{value}" for key, value in kwargs.items())
        row = evaluate_topk_model(
            f"als_{suffix}",
            recommendations,
            train_df=train_df,
            relevant_items=relevant_items,
            catalog=catalog,
        )
        row["rmse"] = np.nan
        row["family"] = "implicit_als"
        rows.append(row)

    comparison_df = pd.DataFrame(rows)
    comparison_df = comparison_df.sort_values(
        by=[f"precision@{DEFAULT_TOP_K}", "rmse"],
        ascending=[False, True],
        na_position="last",
    ).reset_index(drop=True)

    top10_df = comparison_df.head(10).copy()

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(METRICS_DIR / "day2_collaborative_filtering_results.csv", index=False)
    top10_df.to_csv(METRICS_DIR / "day2_collaborative_filtering_top10.csv", index=False)

    print(comparison_df.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
