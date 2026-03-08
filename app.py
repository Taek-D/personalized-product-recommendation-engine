from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.app import (
    MODEL_LABELS,
    build_similar_item_view,
    build_user_profile_view,
    build_user_recommendation_view,
    load_demo_assets,
    recommend_from_seed_ratings,
)
from src.config import PROJECT_ROOT


st.set_page_config(
    page_title="RecSys Lab Demo",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #16302b;
        --teal: #0f766e;
        --sand: #f5efe4;
        --gold: #c77d2b;
        --panel: #fffaf2;
    }
    .hero {
        padding: 1.2rem 1.4rem;
        border-radius: 18px;
        background:
            radial-gradient(circle at top left, rgba(15,118,110,0.18), transparent 38%),
            radial-gradient(circle at bottom right, rgba(199,125,43,0.16), transparent 42%),
            var(--panel);
        border: 1px solid rgba(22,48,43,0.08);
        margin-bottom: 1rem;
    }
    .hero h1 {
        color: var(--ink);
        margin: 0 0 0.35rem 0;
        font-size: 2rem;
        letter-spacing: -0.04em;
    }
    .hero p {
        margin: 0;
        color: #36514b;
        font-size: 1rem;
    }
    .metric-card {
        padding: 0.8rem 1rem;
        border-radius: 14px;
        background: white;
        border: 1px solid rgba(22,48,43,0.08);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_day1_overview() -> dict:
    path = PROJECT_ROOT / "artifacts" / "metrics" / "day1_overview.json"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def title_for_item(item_id: int, items: pd.DataFrame) -> str:
    item = items.loc[items["item_id"] == item_id].iloc[0]
    year = int(item["release_year"]) if pd.notna(item["release_year"]) else "unknown"
    return f"{item['title']} ({year})"


def main() -> None:
    assets = load_demo_assets()
    overview = load_day1_overview()
    items = assets.items
    item_ids = items["item_id"].astype(int).tolist()
    item_label_lookup = {item_id: title_for_item(item_id, items) for item_id in item_ids}

    st.markdown(
        """
        <section class="hero">
            <h1>RecSys Lab: 개인화 상품 추천 엔진</h1>
            <p>Collaborative Filtering, Content-Based, Hybrid 추천을 MovieLens 100K로 비교하고,
            cold-start 전략과 A/B 테스트 설계까지 한 화면에서 확인하는 데모입니다.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    user_ids = sorted(assets.test_df["user_id"].unique())
    default_user_index = user_ids.index(1) if 1 in user_ids else 0

    with st.sidebar:
        st.subheader("탐색 설정")
        user_id = st.selectbox("유저 선택", user_ids, index=default_user_index)
        top_k = st.slider("Top-K", min_value=5, max_value=15, value=10)

        st.divider()
        st.subheader("신규 유저 시뮬레이션")
        seed_item_ids = st.multiselect(
            "좋아한다고 가정할 영화",
            options=item_ids,
            default=[50, 181] if 50 in item_ids and 181 in item_ids else item_ids[:2],
            format_func=lambda item_id: item_label_lookup[item_id],
            max_selections=5,
        )

        seed_rows: list[dict[str, float | int]] = []
        for item_id in seed_item_ids:
            rating = st.slider(
                f"평점: {item_label_lookup[item_id]}",
                min_value=1,
                max_value=5,
                value=5,
                key=f"rating_{item_id}",
            )
            seed_rows.append({"item_id": item_id, "rating": rating})

    observed_seed_ratings = pd.DataFrame(seed_rows)
    recommendation_view = build_user_recommendation_view(assets, user_id, top_k=top_k)
    user_profile_view = build_user_profile_view(assets, user_id, top_n=5)

    tab_eda, tab_ranker, tab_similarity, tab_dashboard, tab_cold_start, tab_ab = st.tabs(
        [
            "EDA",
            "모델별 Top-10",
            "유사 아이템",
            "모델 비교",
            "Cold-Start",
            "A/B 테스트",
        ]
    )

    with tab_eda:
        col1, col2, col3, col4 = st.columns(4)
        dataset_overview = overview.get("dataset_overview", {})
        col1.metric("Ratings", f"{dataset_overview.get('n_ratings', len(assets.bundle.ratings)):,}")
        col2.metric("Users", f"{dataset_overview.get('n_users', assets.bundle.ratings['user_id'].nunique()):,}")
        col3.metric("Items", f"{dataset_overview.get('n_items', assets.bundle.ratings['item_id'].nunique()):,}")
        col4.metric("Sparsity", f"{dataset_overview.get('sparsity', 0.0):.2%}")

        rating_dist = (
            assets.bundle.ratings["rating"].value_counts().sort_index().rename_axis("rating").reset_index(name="count")
        )
        user_counts = assets.bundle.ratings.groupby("user_id")["item_id"].count().reset_index(name="rating_count")
        item_counts = assets.bundle.ratings.groupby("item_id")["user_id"].count().reset_index(name="rating_count")

        col_left, col_right = st.columns(2)
        with col_left:
            st.plotly_chart(
                px.bar(rating_dist, x="rating", y="count", title="평점 분포", color="rating", color_continuous_scale="Tealgrn"),
                use_container_width=True,
            )
        with col_right:
            st.plotly_chart(
                px.histogram(user_counts, x="rating_count", nbins=40, title="유저별 평점 수 분포", color_discrete_sequence=["#c77d2b"]),
                use_container_width=True,
            )

        st.plotly_chart(
            px.histogram(item_counts, x="rating_count", nbins=50, title="아이템별 평점 수 분포", color_discrete_sequence=["#0f766e"]),
            use_container_width=True,
        )

    with tab_ranker:
        col_left, col_right = st.columns([0.9, 1.1])
        with col_left:
            st.markdown(f"### User {user_id} 프로필")
            st.dataframe(
                user_profile_view[["title", "genres", "release_year", "rating"]].rename(
                    columns={
                        "title": "title",
                        "genres": "genres",
                        "release_year": "year",
                        "rating": "rating",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )
        with col_right:
            st.markdown("### 모델별 추천 결과")
            for model_label, frame in recommendation_view.groupby("model", sort=False):
                st.markdown(f"#### {model_label}")
                st.dataframe(
                    frame[["rank", "title", "genres", "release_year"]].rename(columns={"release_year": "year"}),
                    use_container_width=True,
                    hide_index=True,
                )

    with tab_similarity:
        default_item_id = 50 if 50 in item_ids else item_ids[0]
        anchor_item_id = st.selectbox(
            "기준 영화",
            options=item_ids,
            index=item_ids.index(default_item_id),
            format_func=lambda item_id: item_label_lookup[item_id],
            key="anchor_item_id",
        )
        anchor_row = items.loc[items["item_id"] == anchor_item_id].iloc[0]
        st.markdown(
            f"**선택 영화:** {anchor_row['title']}  \n"
            f"장르: {', '.join(anchor_row['genres'])}  \n"
            f"개봉연도: {int(anchor_row['release_year']) if pd.notna(anchor_row['release_year']) else 'unknown'}"
        )
        similar_items = build_similar_item_view(assets, anchor_item_id, top_k=top_k)
        st.dataframe(
            similar_items[["rank", "title", "genres", "release_year", "similarity"]].rename(columns={"release_year": "year"}),
            use_container_width=True,
            hide_index=True,
        )

    with tab_dashboard:
        dashboard_df = assets.model_metrics.copy()
        dashboard_df["model_label"] = dashboard_df["model_name"].map(MODEL_LABELS).fillna(dashboard_df["model_name"])

        scatter = px.scatter(
            dashboard_df,
            x="coverage",
            y="precision@10",
            size="ndcg@10",
            color="model_label",
            hover_data=["map@10", "intra_list_diversity"],
            title="정확도와 Coverage의 trade-off",
        )
        st.plotly_chart(scatter, use_container_width=True)

        if not assets.segment_metrics.empty:
            segment_view = assets.segment_metrics.copy()
            segment_view["model_label"] = segment_view["model_name"].map(MODEL_LABELS).fillna(segment_view["model_name"])
            segment_chart = px.bar(
                segment_view,
                x="segment",
                y="precision@10",
                color="model_label",
                barmode="group",
                title="유저 세그먼트별 Precision@10",
            )
            st.plotly_chart(segment_chart, use_container_width=True)

        st.dataframe(
            dashboard_df[
                [
                    "model_label",
                    "precision@10",
                    "recall@10",
                    "ndcg@10",
                    "map@10",
                    "coverage",
                    "cold_start_item_share",
                ]
            ].sort_values("precision@10", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

    with tab_cold_start:
        col_left, col_right = st.columns([0.95, 1.05])
        with col_left:
            st.markdown("### 신규 유저 추천")
            if observed_seed_ratings.empty:
                st.info("사이드바에서 seed 영화를 1개 이상 선택하면 신규 유저 추천을 계산합니다.")
            else:
                observed_seed_ratings["user_id"] = -1
                seed_recommendations = recommend_from_seed_ratings(
                    assets,
                    observed_seed_ratings[["user_id", "item_id", "rating"]],
                    top_k=top_k,
                )
                st.dataframe(
                    seed_recommendations[["strategy", "rank", "title", "genres", "release_year"]].rename(columns={"release_year": "year"}),
                    use_container_width=True,
                    hide_index=True,
                )

            st.markdown("### 신규 아이템 전략")
            st.dataframe(
                assets.new_item_strategy.rename(columns={"model_name": "model", "recommended_stage": "stage"}),
                use_container_width=True,
                hide_index=True,
            )

        with col_right:
            st.markdown("### 몇 개 평점이 있어야 CF가 의미 있는가")
            transition_df = assets.cold_start_transition.copy()
            if not transition_df.empty:
                line = px.line(
                    transition_df,
                    x="observed_ratings",
                    y=["precision@10", "coverage"],
                    markers=True,
                    title="온보딩 단계별 성능 변화",
                )
                st.plotly_chart(line, use_container_width=True)
                st.dataframe(transition_df, use_container_width=True, hide_index=True)

    with tab_ab:
        st.markdown("### 추천 시스템 A/B 테스트 설계")
        if assets.ab_test_markdown:
            st.markdown(assets.ab_test_markdown)
        else:
            st.warning("`ab_test_design.md` 파일이 아직 없습니다.")

        if not assets.scenario_best_models.empty:
            st.markdown("### 시나리오별 최적 모델")
            scenario_view = assets.scenario_best_models.copy()
            scenario_view["recommended_model"] = scenario_view["recommended_model"].map(MODEL_LABELS).fillna(
                scenario_view["recommended_model"]
            )
            st.dataframe(scenario_view, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
