from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path

import pandas as pd

from src.config import DEFAULT_TOP_K, HIGH_RATING_THRESHOLD, PROJECT_ROOT, RANDOM_SEED
from src.data import build_relevance_sets, load_bundle, random_train_test_split
from src.evaluation.cold_start import recommend_for_new_user
from src.features import (
    build_item_similarity_matrix,
    build_tfidf_item_feature_matrix,
    build_user_profile_matrix,
    explain_user_profile,
    score_user_item_content,
)
from src.models import (
    blend_score_matrices,
    build_explicit_matrix,
    compute_similarity,
    predict_user_based_scores,
    recommend_from_score_matrix,
    recommend_popular_items,
    switch_score_matrices,
    users_below_interaction_threshold,
)


MODEL_LABELS = {
    "popularity_baseline": "Popularity Baseline",
    "cf_user_pearson_k40": "User-CF (pearson, k=40)",
    "content_tfidf_metadata": "Content-Based TF-IDF",
    "weighted_hybrid_alpha_0.85": "Weighted Hybrid (alpha=0.85)",
    "switching_hybrid_lt_15": "Switching Hybrid (<15 -> CB)",
}

MODEL_ORDER = [
    "popularity_baseline",
    "cf_user_pearson_k40",
    "content_tfidf_metadata",
    "weighted_hybrid_alpha_0.85",
    "switching_hybrid_lt_15",
]


@dataclass(slots=True)
class DemoAssets:
    bundle: object
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    relevant_items: dict[int, set[int]]
    items: pd.DataFrame
    item_feature_matrix: pd.DataFrame
    metadata_item_similarity: pd.DataFrame
    item_cf_similarity: pd.DataFrame
    popularity_ranking: list[int]
    model_recommendations: dict[str, dict[int, list[int]]]
    model_metrics: pd.DataFrame
    segment_metrics: pd.DataFrame
    scenario_best_models: pd.DataFrame
    cold_start_transition: pd.DataFrame
    new_item_strategy: pd.DataFrame
    hybrid_summary: dict
    ab_test_markdown: str


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _read_json_if_exists(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _read_text_if_exists(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _build_new_item_strategy(model_metrics: pd.DataFrame) -> pd.DataFrame:
    if model_metrics.empty:
        return pd.DataFrame()

    selected = model_metrics.loc[
        model_metrics["model_name"].isin(
            [
                "content_tfidf_metadata",
                "weighted_hybrid_alpha_0.85",
                "switching_hybrid_lt_15",
                "popularity_baseline",
            ]
        ),
        ["model_name", "precision@10", "coverage", "cold_start_item_share"],
    ].copy()
    if selected.empty:
        return selected

    selected["recommended_stage"] = selected["model_name"].map(
        {
            "content_tfidf_metadata": "metadata-only launch",
            "switching_hybrid_lt_15": "early traction",
            "weighted_hybrid_alpha_0.85": "steady state",
            "popularity_baseline": "fallback baseline",
        }
    )
    selected["model_label"] = selected["model_name"].map(MODEL_LABELS).fillna(selected["model_name"])
    return selected.sort_values("cold_start_item_share", ascending=False).reset_index(drop=True)


@lru_cache(maxsize=1)
def load_demo_assets() -> DemoAssets:
    bundle = load_bundle(download_if_missing=False)
    train_df, test_df = random_train_test_split(bundle.ratings, test_size=0.2, random_state=RANDOM_SEED)
    relevant_items = build_relevance_sets(test_df, min_rating=HIGH_RATING_THRESHOLD)
    items = bundle.items.copy()

    item_feature_matrix = build_tfidf_item_feature_matrix(items)
    metadata_item_similarity = build_item_similarity_matrix(item_feature_matrix)
    item_cf_similarity = compute_similarity(build_explicit_matrix(train_df), axis="item", metric="pearson").fillna(0.0)

    popularity_ranking = train_df.groupby("item_id")["user_id"].count().sort_values(ascending=False).index.astype(int).tolist()
    target_user_ids = sorted(test_df["user_id"].unique())

    popularity_recommendations = recommend_popular_items(train_df, user_ids=target_user_ids, top_k=DEFAULT_TOP_K)

    cf_user_score_matrix, _ = predict_user_based_scores(train_df, k=40, metric="pearson")
    cf_recommendations = recommend_from_score_matrix(
        cf_user_score_matrix,
        train_df,
        user_ids=target_user_ids,
        top_k=DEFAULT_TOP_K,
    )

    content_user_profiles = build_user_profile_matrix(train_df, item_feature_matrix)
    content_score_matrix = score_user_item_content(content_user_profiles, item_feature_matrix)
    content_recommendations = recommend_from_score_matrix(
        content_score_matrix,
        train_df,
        user_ids=[user_id for user_id in target_user_ids if user_id in content_score_matrix.index],
        top_k=DEFAULT_TOP_K,
    )
    for user_id in target_user_ids:
        content_recommendations.setdefault(user_id, [])

    weighted_hybrid_score_matrix = blend_score_matrices(
        cf_user_score_matrix,
        content_score_matrix,
        alpha=0.85,
        normalization="user_minmax",
    )
    weighted_hybrid_recommendations = recommend_from_score_matrix(
        weighted_hybrid_score_matrix,
        train_df,
        user_ids=target_user_ids,
        top_k=DEFAULT_TOP_K,
    )

    switching_users = users_below_interaction_threshold(train_df, threshold=15)
    switching_score_matrix = switch_score_matrices(
        cf_user_score_matrix,
        content_score_matrix,
        secondary_user_ids=switching_users,
    )
    switching_recommendations = recommend_from_score_matrix(
        switching_score_matrix,
        train_df,
        user_ids=target_user_ids,
        top_k=DEFAULT_TOP_K,
    )

    metrics_dir = PROJECT_ROOT / "artifacts" / "metrics"
    model_metrics = _read_csv_if_exists(metrics_dir / "day4_model_comparison.csv")
    segment_metrics = _read_csv_if_exists(metrics_dir / "day4_segment_comparison.csv")
    scenario_best_models = _read_csv_if_exists(metrics_dir / "day4_scenario_best_models.csv")
    cold_start_transition = _read_csv_if_exists(metrics_dir / "day5_cold_start_transition.csv")
    hybrid_summary = _read_json_if_exists(metrics_dir / "day4_hybrid_summary.json")
    ab_test_markdown = _read_text_if_exists(PROJECT_ROOT / "ab_test_design.md")

    new_item_strategy = _build_new_item_strategy(model_metrics)

    return DemoAssets(
        bundle=bundle,
        train_df=train_df,
        test_df=test_df,
        relevant_items=relevant_items,
        items=items,
        item_feature_matrix=item_feature_matrix,
        metadata_item_similarity=metadata_item_similarity,
        item_cf_similarity=item_cf_similarity,
        popularity_ranking=popularity_ranking,
        model_recommendations={
            "popularity_baseline": popularity_recommendations,
            "cf_user_pearson_k40": cf_recommendations,
            "content_tfidf_metadata": content_recommendations,
            "weighted_hybrid_alpha_0.85": weighted_hybrid_recommendations,
            "switching_hybrid_lt_15": switching_recommendations,
        },
        model_metrics=model_metrics,
        segment_metrics=segment_metrics,
        scenario_best_models=scenario_best_models,
        cold_start_transition=cold_start_transition,
        new_item_strategy=new_item_strategy,
        hybrid_summary=hybrid_summary,
        ab_test_markdown=ab_test_markdown,
    )


def enrich_items(item_ids: list[int], items: pd.DataFrame) -> pd.DataFrame:
    if not item_ids:
        return pd.DataFrame(columns=["rank", "item_id", "title", "genres", "release_year"])

    lookup = items.set_index("item_id")
    rows = []
    for rank, item_id in enumerate(item_ids, start=1):
        if item_id not in lookup.index:
            continue
        item = lookup.loc[item_id]
        rows.append(
            {
                "rank": rank,
                "item_id": item_id,
                "title": item["title"],
                "genres": ", ".join(item["genres"]),
                "release_year": item["release_year"],
            }
        )
    return pd.DataFrame(rows)


def build_user_recommendation_view(assets: DemoAssets, user_id: int, *, top_k: int = DEFAULT_TOP_K) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for model_name in MODEL_ORDER:
        recommendations = assets.model_recommendations.get(model_name, {}).get(user_id, [])[:top_k]
        frame = enrich_items(recommendations, assets.items)
        if frame.empty:
            continue
        frame.insert(0, "model", MODEL_LABELS.get(model_name, model_name))
        frames.append(frame)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_similar_item_view(assets: DemoAssets, item_id: int, *, top_k: int = DEFAULT_TOP_K) -> pd.DataFrame:
    if item_id not in assets.metadata_item_similarity.index:
        return pd.DataFrame(columns=["rank", "item_id", "title", "genres", "release_year", "similarity"])

    similar = (
        assets.metadata_item_similarity.loc[item_id]
        .drop(labels=[item_id], errors="ignore")
        .sort_values(ascending=False)
        .head(top_k)
        .rename("similarity")
        .reset_index()
        .rename(columns={"index": "item_id"})
    )
    enriched = enrich_items(similar["item_id"].astype(int).tolist(), assets.items)
    if enriched.empty:
        return enriched
    enriched = enriched.merge(similar, on="item_id", how="left")
    return enriched[["rank", "item_id", "title", "genres", "release_year", "similarity"]]


def build_user_profile_view(assets: DemoAssets, user_id: int, *, top_n: int = 5) -> pd.DataFrame:
    return explain_user_profile(user_id, assets.train_df, assets.items, top_n=top_n)


def recommend_from_seed_ratings(
    assets: DemoAssets,
    observed_ratings: pd.DataFrame,
    *,
    top_k: int = DEFAULT_TOP_K,
    cf_switch_threshold: int = 5,
    hybrid_alpha: float = 0.85,
) -> pd.DataFrame:
    bundle = recommend_for_new_user(
        observed_ratings=observed_ratings,
        popularity_ranking=assets.popularity_ranking,
        item_feature_matrix=assets.item_feature_matrix,
        item_similarity=assets.item_cf_similarity,
        top_k=top_k,
        cf_switch_threshold=cf_switch_threshold,
        hybrid_alpha=hybrid_alpha,
    )
    frame = enrich_items(bundle.recommendations, assets.items)
    if frame.empty:
        return frame
    frame.insert(0, "strategy", bundle.strategy)
    return frame
