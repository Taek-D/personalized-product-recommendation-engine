from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.app import MODEL_LABELS, load_demo_assets
from src.config import PROJECT_ROOT


st.set_page_config(page_title="Methodology", layout="wide")
assets = load_demo_assets()

st.title("Methodology")
st.caption("실험 구조, 세그먼트별 결과, 핵심 아티팩트를 빠르게 훑는 페이지입니다.")

col1, col2 = st.columns(2)
with col1:
    if not assets.scenario_best_models.empty:
        scenario_view = assets.scenario_best_models.copy()
        scenario_view["recommended_model"] = scenario_view["recommended_model"].map(MODEL_LABELS).fillna(
            scenario_view["recommended_model"]
        )
        st.subheader("Scenario Best Model")
        st.dataframe(scenario_view, use_container_width=True, hide_index=True)

with col2:
    if not assets.segment_metrics.empty:
        segment_view = assets.segment_metrics.copy()
        segment_view["model_label"] = segment_view["model_name"].map(MODEL_LABELS).fillna(segment_view["model_name"])
        fig = px.bar(
            segment_view,
            x="segment",
            y="precision@10",
            color="model_label",
            barmode="group",
            title="유저 세그먼트별 Precision@10",
        )
        st.plotly_chart(fig, use_container_width=True)

st.subheader("Artifacts")
figure_candidates = [
    PROJECT_ROOT / "artifacts" / "figures" / "day4_model_comparison_dashboard.png",
    PROJECT_ROOT / "artifacts" / "figures" / "day4_model_tradeoff.png",
    PROJECT_ROOT / "artifacts" / "figures" / "day5_cold_start_transition.png",
]

for figure_path in figure_candidates:
    if figure_path.exists():
        st.image(str(figure_path), caption=str(figure_path.relative_to(PROJECT_ROOT)), use_container_width=True)
