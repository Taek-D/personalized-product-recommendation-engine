# AGENTS.md

## 적용 범위
이 파일의 지침은 저장소 루트와 그 하위 전체에 적용됩니다.

## 프로젝트 한줄 정의
- 프로젝트명: **RecSys Lab: 개인화 상품 추천 엔진**
- 목표: **MovieLens 100K** 기반으로 Collaborative Filtering, Content-Based Filtering, Hybrid 추천을 비교하고, **Top-10 추천 품질**과 **비즈니스 해석 가능성**을 함께 보여주는 포트폴리오 프로젝트를 완성한다.
- 핵심 문장: **"Collaborative Filtering + Content-Based + Hybrid 3가지 접근법으로 Top-10 추천 성능을 비교하고 Precision@10을 개선한다."**

## 진실의 원천(Source of Truth)
작업 우선순위는 아래 문서를 기준으로 맞춘다.
1. `## 🎯 프로젝트 3 개인화 상품 추천 엔진.txt`
2. `task.md`
3. 이 `AGENTS.md`

충돌 시에는 원본 기획 문서의 방향을 유지하고, `task.md`는 실행 가능한 작업 단위로 구체화하는 용도로 사용한다.

## 필수 산출물
아래 결과물이 최종적으로 보이도록 작업한다.
- `01_eda.ipynb`
- `02_collaborative_filtering.ipynb`
- `03_content_based.ipynb`
- `04_hybrid.ipynb`
- `05_model_comparison.ipynb`
- `06_cold_start.ipynb`
- `ab_test_design.md`
- `app.py`
- `pages/`
- `README.md`
- `requirements.txt`

## 권장 저장소 구조
프로젝트를 비우지 말고, 가능하면 아래 구조를 기준으로 확장한다.

```text
.
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_collaborative_filtering.ipynb
│   ├── 03_content_based.ipynb
│   ├── 04_hybrid.ipynb
│   ├── 05_model_comparison.ipynb
│   └── 06_cold_start.ipynb
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── evaluation/
│   └── app/
├── artifacts/
│   ├── figures/
│   └── metrics/
├── pages/
├── app.py
├── ab_test_design.md
├── README.md
├── requirements.txt
└── task.md
```

## 작업 원칙
- 코드는 재사용 가능한 로직을 `src/`에 두고, 노트북은 실험/설명 중심으로 유지한다.
- 원본 데이터 경로를 **하드코딩하지 않는다**. 절대경로(특히 Windows 경로) 대신 상대경로/설정값을 사용한다.
- 평가 실험에는 가능하면 **random seed**, **데이터 분할 방식**, **평가 기준**을 명시한다.
- Top-K 추천 평가는 반드시 **학습에서 이미 본 아이템을 제외**하고 계산한다.
- 모든 모델 비교에는 최소 1개의 **베이스라인(인기도, 전체 평균, 유저 평균 등)** 을 포함한다.
- 측정된 결과만 기록하고, 아직 확인되지 않은 수치(`0.XX`, `+XX%`)는 placeholder로 남긴다.
- 결과 표/차트/지표는 가능하면 `artifacts/metrics`, `artifacts/figures`에 저장한다.
- 문서는 한국어 중심으로 작성해도 되지만, 코드 식별자와 파일명은 일관된 영어 기반을 유지한다.

## 모델/분석 범위
최소 범위는 아래를 포함한다.
- **Collaborative Filtering**
  - User-Based CF
  - Item-Based CF
  - SVD / SVD++ / NMF
  - ALS(implicit)
- **Content-Based Filtering**
  - 장르 기반 특성
  - 개봉연도/구간화 특성
  - 가능하면 태그 기반 TF-IDF
- **Hybrid**
  - Weighted Hybrid
  - Switching Hybrid
  - Cold-start 대응 전략

## 핵심 평가 지표
최소한 아래 지표는 추적한다.
- RMSE
- Precision@K
- Recall@K
- NDCG@K
- MAP
- Coverage
- Intra-list Diversity
- Cold-start 그룹 성능 비교

## Streamlit 앱 요구사항
앱에는 가능하면 아래 구성이 반영되어야 한다.
- 사이드바: 유저 선택 / 신규 유저 시뮬레이션
- Tab 1: EDA
- Tab 2: 모델별 Top-10 추천 비교
- Tab 3: 아이템 기반 유사 추천
- Tab 4: 모델 비교 대시보드
- Tab 5: Cold-start 시뮬레이션
- Tab 6: A/B 테스트 설계 요약

## 문서화 요구사항
최종 문서에는 아래 내용이 포함되도록 한다.
- 왜 MovieLens를 사용했는지
- 왜 이 분석/추천 방식을 선택했는지
- 오프라인 지표와 온라인 비즈니스 지표의 연결 논리
- 한계점과 다음 단계
- 포트폴리오용 Problem / Solution / Impact / Learning 요약

## 에이전트 작업 규칙
- 큰 작업을 시작하기 전에 `task.md`를 먼저 읽고, 해당 섹션의 체크박스/상태를 갱신한다.
- 아직 구현되지 않은 항목은 추정 완료 처리하지 않는다.
- 새로운 파일을 만들 때는 되도록 위의 권장 구조를 따른다.
- 실험 결과를 바꾸는 변경을 했다면, 관련 지표/문서/체크리스트도 함께 갱신한다.
- 불필요한 대용량 산출물이나 원본 데이터 복제본은 저장소 루트에 직접 두지 않는다.

## 완료 기준(Definition of Done)
최종적으로 아래 체크리스트를 만족하는 방향으로 작업한다.
- README에 프로젝트 개요, 실행 방법, 결과 요약이 있다.
- 핵심 시각화 스크린샷 3~4장이 정리되어 있다.
- `requirements.txt`가 있다.
- Streamlit 앱이 로컬 또는 배포 환경에서 동작한다.
- 모델 비교 결과와 cold-start 해석이 문서화되어 있다.
- A/B 테스트 설계 문서가 있다.
- 포트폴리오용 핵심 메시지(Problem / Solution / Impact / Learning)가 정리되어 있다.
