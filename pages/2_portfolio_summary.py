from __future__ import annotations

import streamlit as st

from src.app import MODEL_LABELS, load_demo_assets


st.set_page_config(page_title="Portfolio Summary", layout="centered")
assets = load_demo_assets()

model_metrics = assets.model_metrics.set_index("model_name") if not assets.model_metrics.empty else None

weighted_precision = float(model_metrics.loc["weighted_hybrid_alpha_0.85", "precision@10"]) if model_metrics is not None else 0.0
popularity_precision = float(model_metrics.loc["popularity_baseline", "precision@10"]) if model_metrics is not None else 0.0
precision_uplift = ((weighted_precision / popularity_precision) - 1.0) * 100 if popularity_precision else 0.0

st.title("Portfolio Summary")
st.markdown(
    f"""
### Problem
MovieLens 100K 환경에서 인기도 기반 추천만으로는 long-tail 아이템 노출과 개인화 수준이 제한됩니다.

### Solution
User-CF, Content-Based TF-IDF, Weighted/Switching Hybrid를 비교하고, 신규 유저와 신규 아이템에는 별도 cold-start 전략을 붙였습니다.

### Impact
- {MODEL_LABELS['weighted_hybrid_alpha_0.85']}의 Precision@10은 `{weighted_precision:.4f}`입니다.
- 같은 기준에서 popularity baseline 대비 Precision@10이 `{precision_uplift:.1f}%` 높았습니다.
- Switching Hybrid는 CF 대비 coverage를 늘리면서 sparse profile user 대응 옵션으로 해석할 수 있습니다.

### Learning
- RMSE가 낮다고 Top-K ranking이 좋은 것은 아니었습니다.
- Content-Based는 정확도보다 cold-start item 노출과 coverage에서 더 큰 가치를 보였습니다.
- offline metric을 바로 비즈니스 성과로 해석하지 않기 위해 A/B 테스트 설계를 함께 문서화했습니다.
"""
)

if assets.cold_start_transition is not None and not assets.cold_start_transition.empty:
    st.subheader("Cold-Start Threshold")
    st.dataframe(assets.cold_start_transition, use_container_width=True, hide_index=True)
