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
        padding: 1.6rem 1.8rem;
        border-radius: 18px;
        background:
            radial-gradient(circle at top left, rgba(15,118,110,0.18), transparent 38%),
            radial-gradient(circle at bottom right, rgba(199,125,43,0.16), transparent 42%),
            var(--panel);
        border: 1px solid rgba(22,48,43,0.08);
        margin-bottom: 0.6rem;
    }
    .hero h1 {
        color: var(--ink);
        margin: 0 0 0.25rem 0;
        font-size: 2rem;
        letter-spacing: -0.04em;
    }
    .hero .tagline {
        margin: 0 0 0.8rem 0;
        color: #36514b;
        font-size: 1.05rem;
        font-weight: 500;
    }
    .hero .story {
        margin: 0;
        color: #4a6b63;
        font-size: 0.92rem;
        line-height: 1.6;
    }
    .hero .story strong {
        color: var(--ink);
    }
    .metric-row {
        display: flex;
        gap: 0.8rem;
        margin-top: 1rem;
    }
    .metric-card {
        flex: 1;
        padding: 0.9rem 1rem;
        border-radius: 14px;
        background: white;
        border: 1px solid rgba(22,48,43,0.08);
        text-align: center;
    }
    .metric-card .value {
        font-size: 1.6rem;
        font-weight: 700;
        color: var(--teal);
        margin: 0;
    }
    .metric-card .label {
        font-size: 0.78rem;
        color: #6b8f87;
        margin: 0.2rem 0 0 0;
    }
    .metric-card .sub {
        font-size: 0.72rem;
        color: var(--gold);
        margin: 0.15rem 0 0 0;
        font-weight: 600;
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

    # ── Hero: 프로젝트 스토리 + 핵심 성과 ──
    st.markdown(
        """
        <section class="hero">
            <h1>RecSys Lab: 개인화 상품 추천 엔진</h1>
            <p class="tagline">인기도 추천의 한계를 넘어, 유저 취향에 맞는 Top-10을 찾아가는 과정</p>
            <p class="story">
                <strong>Problem</strong> — 모든 유저에게 같은 인기 영화만 추천하면 취향 반영이 안 되고, 신규 유저/아이템은 추천 자체가 불가능합니다.<br>
                <strong>Approach</strong> — 6종의 추천 알고리즘(CF, Content-Based, Hybrid)을 동일 기준으로 비교하고, 신규 유저 대응 전략과 A/B 테스트 설계까지 연결했습니다.<br>
                <strong>Result</strong> — Weighted Hybrid가 인기도 추천 대비 <strong>Top-10 정확도 85.4% 개선</strong>, 추천 범위 6.2배 확대를 달성했습니다.
            </p>
            <div class="metric-row">
                <div class="metric-card">
                    <p class="value">+85.4%</p>
                    <p class="label">Precision@10 개선</p>
                    <p class="sub">vs 인기도 추천</p>
                </div>
                <div class="metric-card">
                    <p class="value">+103.4%</p>
                    <p class="label">NDCG@10 개선</p>
                    <p class="sub">랭킹 품질 2배</p>
                </div>
                <div class="metric-card">
                    <p class="value">6.2x</p>
                    <p class="label">Coverage 확대</p>
                    <p class="sub">2.9% → 18.1%</p>
                </div>
                <div class="metric-card">
                    <p class="value">6종</p>
                    <p class="label">모델 비교</p>
                    <p class="sub">CF · CB · Hybrid</p>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    # ── Sidebar ──
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

    # ── Tabs ──
    tab_result, tab_recommend, tab_coldstart, tab_ab, tab_similarity, tab_eda = st.tabs(
        [
            "📈 핵심 결과",
            "🎯 추천 체험",
            "🆕 신규 유저 대응",
            "🧪 A/B 테스트 설계",
            "🔍 유사 아이템",
            "📊 데이터 개요",
        ]
    )

    # ── Tab 1: 핵심 결과 (구 모델 비교) ──
    with tab_result:
        st.markdown("### 모델별 성능 비교")
        st.caption("같은 평가 기준(Precision@10, NDCG@10, Coverage)으로 6종 모델을 비교한 결과입니다.")

        dashboard_df = assets.model_metrics.copy()
        dashboard_df["model_label"] = dashboard_df["model_name"].map(MODEL_LABELS).fillna(dashboard_df["model_name"])

        col_chart, col_table = st.columns([1.1, 0.9])
        with col_chart:
            scatter = px.scatter(
                dashboard_df,
                x="coverage",
                y="precision@10",
                size="ndcg@10",
                color="model_label",
                hover_data=["map@10", "intra_list_diversity"],
                title="정확도 vs 추천 범위 — 어떤 모델이 두 마리 토끼를 잡는가?",
            )
            scatter.update_layout(
                xaxis_title="Coverage (추천 범위)",
                yaxis_title="Precision@10 (정확도)",
            )
            st.plotly_chart(scatter, use_container_width=True)

        with col_table:
            st.dataframe(
                dashboard_df[
                    [
                        "model_label",
                        "precision@10",
                        "recall@10",
                        "ndcg@10",
                        "map@10",
                        "coverage",
                    ]
                ]
                .sort_values("precision@10", ascending=False)
                .rename(columns={"model_label": "모델"}),
                use_container_width=True,
                hide_index=True,
            )

        if not assets.segment_metrics.empty:
            st.markdown("### 유저 활동량별 성능 차이")
            st.caption("활발한 유저, 보통 유저, 신규 유저 — 누구에게 어떤 모델이 강한가?")
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

        st.markdown("---")
        st.markdown(
            "**핵심 발견**: RMSE가 낮은 모델(SVD++)이 랭킹 품질에서는 User-CF보다 낮았습니다. "
            "**평점 예측 정확도 ≠ 추천 랭킹 품질** — 비즈니스 목표에 맞는 평가 지표 선택이 중요합니다."
        )

    # ── Tab 2: 추천 체험 ──
    with tab_recommend:
        st.markdown("### 유저별 추천 결과 비교")
        st.caption("좌측에서 유저를 선택하면 각 모델이 그 유저에게 추천하는 영화를 비교할 수 있습니다.")

        col_left, col_right = st.columns([0.9, 1.1])
        with col_left:
            st.markdown(f"#### User {user_id}의 시청 이력")
            st.dataframe(
                user_profile_view[["title", "genres", "release_year", "rating"]].rename(
                    columns={"release_year": "year"}
                ),
                use_container_width=True,
                hide_index=True,
            )
        with col_right:
            st.markdown("#### 모델별 Top-10 추천")
            for model_label, frame in recommendation_view.groupby("model", sort=False):
                st.markdown(f"**{model_label}**")
                st.dataframe(
                    frame[["rank", "title", "genres", "release_year"]].rename(columns={"release_year": "year"}),
                    use_container_width=True,
                    hide_index=True,
                )

    # ── Tab 3: 신규 유저 대응 ──
    with tab_coldstart:
        st.markdown("### 신규 유저, 몇 개 평점이 있어야 개인화가 가능한가?")
        st.caption("평점이 0개인 완전 신규 유저부터 15개까지 — 단계별 최적 전략을 시뮬레이션합니다.")

        col_left, col_right = st.columns([0.95, 1.05])
        with col_right:
            transition_df = assets.cold_start_transition.copy()
            if not transition_df.empty:
                line = px.line(
                    transition_df,
                    x="observed_ratings",
                    y=["precision@10", "coverage"],
                    markers=True,
                    title="평점 수에 따른 추천 품질 변화",
                )
                line.update_layout(
                    xaxis_title="보유 평점 수",
                    yaxis_title="성능",
                )
                st.plotly_chart(line, use_container_width=True)
                st.dataframe(transition_df, use_container_width=True, hide_index=True)

        with col_left:
            st.markdown("#### 사이드바에서 영화를 골라 신규 유저 체험")
            if observed_seed_ratings.empty:
                st.info("사이드바에서 좋아하는 영화를 1개 이상 선택하면 신규 유저 추천을 계산합니다.")
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

            st.markdown("#### 신규 아이템은 어떤 모델이 유리한가?")
            st.dataframe(
                assets.new_item_strategy.rename(columns={"model_name": "model", "recommended_stage": "stage"}),
                use_container_width=True,
                hide_index=True,
            )

        st.markdown("---")
        st.markdown(
            "**핵심 발견**: 약 **10~15개** 평점이 쌓이면 Hybrid가 인기도 추천보다 높은 정확도를 회복합니다. "
            "그 전까지는 Content-Based fallback이 추천 다양성을 확보하는 역할을 합니다."
        )

    # ── Tab 4: A/B 테스트 설계 ──
    with tab_ab:
        st.markdown("### Offline 결과를 Online 실험으로 연결하는 설계")
        st.caption("모델 성능표로 끝나지 않고, 실제 서비스에서 어떻게 검증할 것인지까지 설계했습니다.")

        if assets.ab_test_markdown:
            st.markdown(assets.ab_test_markdown)
        else:
            st.warning("`ab_test_design.md` 파일이 아직 없습니다.")

        if not assets.scenario_best_models.empty:
            st.markdown("### 시나리오별 최적 모델 추천")
            scenario_view = assets.scenario_best_models.copy()
            scenario_view["recommended_model"] = scenario_view["recommended_model"].map(MODEL_LABELS).fillna(
                scenario_view["recommended_model"]
            )
            st.dataframe(scenario_view, use_container_width=True, hide_index=True)

    # ── Tab 5: 유사 아이템 ──
    with tab_similarity:
        st.markdown("### 이 영화를 좋아하면? — 아이템 기반 유사 추천")
        st.caption("Content-Based 유사도(TF-IDF metadata)로 계산한 유사 아이템입니다.")

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

    # ── Tab 6: 데이터 개요 ──
    with tab_eda:
        st.markdown("### MovieLens 100K 데이터 탐색")
        st.caption("943명의 유저가 1,682개 영화에 남긴 10만 건의 평점 데이터입니다.")

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
                px.histogram(user_counts, x="rating_count", nbins=40, title="유저별 평점 수 분포 (Long-tail)", color_discrete_sequence=["#c77d2b"]),
                use_container_width=True,
            )

        st.plotly_chart(
            px.histogram(item_counts, x="rating_count", nbins=50, title="아이템별 평점 수 분포", color_discrete_sequence=["#0f766e"]),
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
