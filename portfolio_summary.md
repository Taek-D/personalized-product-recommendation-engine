# Portfolio Summary

## Problem

인기도 기반 추천만으로는 개인화 수준이 낮고, 신규 아이템은 충분한 노출을 얻기 어렵다.

## Solution

- Collaborative Filtering, Content-Based TF-IDF, Weighted/Switching Hybrid를 같은 지표 체계에서 비교
- Top-10 랭킹 품질과 coverage를 동시에 확인
- 신규 유저와 신규 아이템에 대한 cold-start 전환 로직을 별도로 설계
- offline 결과를 online 실험으로 연결하는 A/B 테스트 문서까지 작성

## Impact

- Weighted Hybrid: `Precision@10 = 0.2497`, `NDCG@10 = 0.3571`
- popularity baseline 대비 Precision@10 `+85.4%`
- Switching Hybrid는 User-CF 대비 coverage `+21.8%`
- 신규 유저 시뮬레이션에서는 약 `15개` 평점 이후 hybrid가 popularity 단계보다 높은 precision을 회복

## Learning

- RMSE가 낮아도 Top-K ranking이 좋아지는 것은 아니었다.
- Content-Based는 정확도보다 cold-start item exposure와 coverage에서 더 큰 의미가 있었다.
- 추천 시스템 포트폴리오에서는 모델 성능표만이 아니라 rollout 기준과 experiment design까지 같이 보여줘야 설득력이 높다.

## Tag

- RECOMMENDATION SYSTEM
- COLLABORATIVE FILTERING
- CONTENT-BASED FILTERING
- HYBRID RECOMMENDER
- STREAMLIT

## Meta

- Tech Stack: Python, pandas, scikit-learn, scikit-surprise, implicit, Streamlit
- Glow Color: teal-gold
