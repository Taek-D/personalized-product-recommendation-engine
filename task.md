# task.md

## 프로젝트 목표
MovieLens 100K 데이터를 활용해 **Collaborative Filtering / Content-Based / Hybrid** 추천 모델을 구현하고 비교한다.  
최종적으로는 **Top-10 추천 품질**, **cold-start 대응 전략**, **A/B 테스트 설계**, **Streamlit 데모 앱**까지 포함한 포트폴리오형 추천 시스템 프로젝트를 완성한다.

## 핵심 성공 기준
- [x] MovieLens 100K 데이터 로딩 및 전처리 파이프라인이 재현 가능하다.
- [x] 추천 성능 비교에 필요한 평가 프레임워크(RMSE, Precision@K, Recall@K, NDCG@K, MAP)가 준비되어 있다.
- [x] 최소 1개 이상의 Collaborative Filtering 계열 모델과 1개 이상의 Content-Based 모델이 구현되어 있다.
- [x] Hybrid 전략으로 기존 베이스라인 대비 개선 여부를 확인할 수 있다.
- [x] Cold-start 대응 전략과 한계가 문서화되어 있다.
- [x] Streamlit 앱으로 결과를 시연할 수 있다.
- [x] README / 포트폴리오 요약 / A/B 테스트 설계 문서가 준비되어 있다.

## 현재 상태
- 상태: **Mostly Complete**
- 원본 기획 문서: `## 🎯 프로젝트 3 개인화 상품 추천 엔진.txt`
- 메모: 최종 성능 수치(`0.XX`, `+XX%`)는 실험 후 확정한다.
- 진행 메모: **Phase 0 완료, Day 1 실행 검증 완료, Day 2 전체 CF 비교(메모리 기반 + Surprise + ALS) 완료, Day 5~7 산출물 추가 완료, 외부 배포 URL/실제 Notion 업로드만 미실행**

---

## Phase 0. 프로젝트 부트스트랩
- [x] 기본 폴더 구조 생성 (`data/`, `notebooks/`, `src/`, `artifacts/`, `pages/`)
- [x] `requirements.txt` 초안 작성
- [x] 데이터 다운로드/적재 방식 결정
- [x] 공통 random seed / 경로 설정 방식 정의
- [x] 베이스라인 및 평가 지표 기록 위치 결정

### 부트스트랩 산출물
- `requirements.txt`
- `data/README.md`
- `src/config.py`
- `notebooks/01_eda.ipynb` (초기 뼈대)

### 완료 기준
- [x] 저장소 구조가 이후 실험을 진행할 수 있을 정도로 준비됨
- [x] 환경 구성 및 의존성 목록이 명시됨

---

## Day 1. 데이터 탐색 + 전처리
**예상 시간:** 4~5h  
**산출물:** `01_eda.ipynb`

### 작업 항목
- [x] MovieLens 100K 데이터 다운로드 및 기본 구조 확인
- [x] 평점 분포 분석 (1~5점, 평균, 중앙값)
- [x] 유저당 평점 수 분포 분석
- [x] 아이템당 평점 수 분포 분석
- [x] 희소성(sparsity) 계산
- [x] 시간에 따른 평점 트렌드 확인
- [x] User-Item Matrix 구성
- [x] Explicit / Implicit 피드백 버전 분리
- [x] Cold-start 유저 / 아이템 식별 기준 정의
- [x] Train/Test 분할 전략 정의 (시간 기준 또는 랜덤)
- [x] 평가 지표 정의 (RMSE, Precision@K, Recall@K, NDCG@K, MAP)
- [x] 베이스라인 정의 (전체 평균, 유저 평균, 인기도 기반)
- [x] K 값 후보 설정 (5, 10, 20)

### 현재 준비된 지원 파일
- `src/data/movielens.py`: 다운로드, 로딩, user-item matrix, implicit feedback, sparsity, split helper
- `src/evaluation/metrics.py`: Precision@K, Recall@K, NDCG@K, MAP, Coverage, Intra-list Diversity
- `src/models/baseline.py`: 전체 평균, 유저 평균, 인기도 기반 baseline helper
- `notebooks/01_eda.ipynb`: Day 1 분석 노트북
- `notebooks/01_eda.executed.ipynb`: 실제 실행 검증된 Day 1 노트북
- `artifacts/metrics/day1_overview.json`: 데이터 개요 및 baseline 요약

### 완료 기준
- [x] EDA 노트북에서 데이터 구조와 품질을 설명할 수 있음
- [x] 평가 프레임워크와 베이스라인이 다음 단계에서 재사용 가능함

---

## Day 2. Collaborative Filtering
**예상 시간:** 4~5h  
**산출물:** `02_collaborative_filtering.ipynb`

### 작업 항목
- [x] Memory-Based User-CF 구현
- [x] Memory-Based Item-CF 구현
- [x] 유사도 함수 비교 (코사인, 피어슨, 자카드)
- [x] 이웃 수(K) 튜닝
- [x] Surprise 기반 SVD 구현
- [x] SVD++ 실험
- [x] NMF 실험
- [x] Latent factor 수 튜닝 (10, 20, 50, 100)
- [x] 정규화 파라미터 튜닝
- [x] implicit 기반 ALS 실험
- [x] CF 계열 모델 비교 (RMSE, Precision@10)

### 현재 준비된 지원 파일
- `src/models/collaborative.py`: user/item CF score matrix, similarity 계산, Surprise/ALS wrapper
- `notebooks/02_collaborative_filtering.ipynb`: Day 2 실험 노트북
- `notebooks/02_collaborative_filtering.executed.ipynb`: 실제 실행 검증된 Day 2 노트북
- `artifacts/metrics/day2_memory_cf_results.csv`: 메모리 기반 CF + baseline 비교 결과
- `artifacts/metrics/day2_memory_cf_top10.csv`: Precision@10 기준 상위 결과
- `artifacts/metrics/day2_collaborative_filtering_results.csv`: baseline / memory CF / Surprise / ALS 전체 비교표
- `artifacts/metrics/day2_collaborative_filtering_top10.csv`: Day 2 전체 비교 Top-10 리더보드
- `scripts/generate_day2_artifacts.py`: Day 2 결과 재생성 스크립트

### 완료 기준
- [x] 최소 3개 이상의 CF 계열 모델 성능이 비교표로 정리됨
- [x] 베이스라인 대비 장단점을 설명할 수 있음

---

## Day 3. Content-Based Filtering
**예상 시간:** 4~5h  
**산출물:** `03_content_based.ipynb`

### 작업 항목
- [x] 장르 One-hot / Multi-hot 인코딩
- [x] 개봉연도 또는 decade 특성 생성
- [x] 태그 데이터 사용 가능 여부 확인
- [x] TF-IDF 기반 아이템 특성 벡터 실험
- [x] 아이템 프로필 벡터 생성
- [x] 유저가 높게 평가한 아이템 기반 유저 프로필 생성
- [x] 유저 프로필 vs 아이템 프로필 유사도 계산
- [x] 코사인 유사도 기반 추천 구현
- [x] Precision@K / Recall@K 평가
- [x] Cold-start 아이템 관점에서 CF 대비 비교
- [x] Intra-list Diversity 분석
- [x] Coverage 분석

### 현재 준비된 지원 파일
- `src/features/content_based.py`: genre/decade 기반 item feature, user profile, cosine similarity 추천 helper
- `notebooks/03_content_based.ipynb`: Day 3 분석 노트북
- `notebooks/03_content_based.executed.ipynb`: 실제 실행 검증된 Day 3 노트북
- `artifacts/metrics/day3_content_based_comparison.csv`: content-based vs popularity baseline 비교
- `artifacts/metrics/day3_content_based_summary.json`: Day 3 요약 지표

### 완료 기준
- [x] Content-Based 추천 로직이 작동함
- [x] 정확도 외에 다양성과 커버리지 관점 비교가 가능함

---

## Day 4. Hybrid + 선택적 Deep Learning 접근
**예상 시간:** 4~5h  
**산출물:** `04_hybrid.ipynb`, `05_model_comparison.ipynb`

### 작업 항목
- [x] Weighted Hybrid 구현 (CF score × α + CB score × β)
- [x] Switching Hybrid 구현 (Cold-start → CB, 그 외 → CF)
- [x] α, β 탐색 또는 Grid Search
- [x] Feature Augmentation 가능성 검토
- [x] 전체 모델 비교표 작성
- [x] 시나리오별 최적 모델 정리 (일반 유저 / 신규 유저 / 신규 아이템)
- [x] 정확도 vs 다양성 trade-off 분석
- [ ] (선택) Embedding 기반 Neural CF 검토 또는 실험

### 현재 준비된 지원 파일
- `src/models/hybrid.py`: score alignment, normalization, weighted blend, switching helper
- `src/features/content_based.py`: metadata TF-IDF item feature builder 추가
- `notebooks/04_hybrid.ipynb`: weighted / switching hybrid 실험 노트북
- `notebooks/05_model_comparison.ipynb`: 최종 모델 비교 및 시나리오 정리 노트북
- `artifacts/metrics/day4_hybrid_grid_search.csv`: α 탐색 결과
- `artifacts/metrics/day4_switching_grid_search.csv`: switching threshold 탐색 결과
- `artifacts/metrics/day4_model_comparison.csv`: baseline / CF / CB / hybrid 비교표
- `artifacts/metrics/day4_segment_comparison.csv`: warm/power user segment 비교표
- `artifacts/metrics/day4_scenario_best_models.csv`: 시나리오별 추천 모델 요약
- `artifacts/figures/day4_hybrid_grid.png`: 하이브리드 탐색 시각화
- `artifacts/figures/day4_model_comparison_dashboard.png`: 모델 비교 대시보드
- `artifacts/figures/day4_model_tradeoff.png`: accuracy vs coverage trade-off 시각화

### 완료 기준
- [x] Hybrid 모델의 적용 시나리오를 설명할 수 있음
- [x] 최종 비교표로 어떤 모델을 언제 써야 하는지 제시 가능함

---

## Day 5. Cold-Start 해결 + A/B 테스트 설계
**예상 시간:** 4~5h  
**산출물:** `06_cold_start.ipynb`, `ab_test_design.md`

### 작업 항목
- [x] 신규 유저 cold-start 시나리오 정의
- [x] 인기도 기반 → CF 전환 시뮬레이션
- [x] 신규 아이템 cold-start 시나리오 정의
- [x] Content-Based → Hybrid 전환 조건 정리
- [x] Cold-start 유저 그룹 vs 기존 유저 그룹 성능 비교
- [x] "몇 개 평점이 있어야 CF가 의미 있는가" 실험
- [x] A/B 테스트 가설 정의
- [x] 유저 단위 무작위 배정 방식 설계
- [x] Primary Metric 정의 (CTR, 평균 평점 등)
- [x] Guardrail Metric 정의 (다양성, 커버리지)
- [x] Sample Size / 실험 기간 산정 방식 문서화
- [x] 오프라인 지표와 온라인 지표 연결 논리 정리

### 현재 준비된 지원 파일
- `src/evaluation/cold_start.py`: 신규 유저 score 생성, popularity/content/hybrid 전환 로직, threshold simulation helper
- `scripts/generate_day5_artifacts.py`: Day 5 cold-start CSV/figure 재생성 스크립트
- `notebooks/06_cold_start.ipynb`: Day 5 분석 노트북
- `notebooks/06_cold_start.executed.ipynb`: 실행 검증된 Day 5 노트북
- `artifacts/metrics/day5_cold_start_transition.csv`: observed rating 수별 cold-start 성능 곡선
- `artifacts/metrics/day5_new_item_strategy.csv`: 신규 아이템 단계별 추천 전략 비교
- `artifacts/metrics/day5_cold_start_summary.json`: 전환 임계값 요약
- `artifacts/figures/day5_cold_start_transition.png`: cold-start 전환 시각화
- `ab_test_design.md`: A/B 테스트 설계 문서

### 완료 기준
- [x] Cold-start 대응 전략이 수치 또는 사례와 함께 정리됨
- [x] A/B 테스트 설계 문서가 비즈니스 관점에서 읽힐 정도로 완성됨

---

## Day 6. Streamlit 앱 개발
**예상 시간:** 4~5h  
**산출물:** `app.py`, `pages/`

### 작업 항목
- [x] 앱 기본 구조 설계
- [x] 사이드바: 유저 선택 / 신규 유저 시뮬레이션 UI 구현
- [x] Tab 1: EDA 시각화 연결
- [x] Tab 2: 모델별 Top-10 추천 비교 UI 구현
- [x] Tab 3: 아이템 기반 유사 추천 UI 구현
- [x] Tab 4: 모델 비교 대시보드 구현
- [x] Tab 5: Cold-start 시뮬레이션 UI 구현
- [x] Tab 6: A/B 테스트 설계 요약 UI 구현
- [x] 배포 전 로컬 테스트
- [x] Streamlit Cloud 배포 준비

### 현재 준비된 지원 파일
- `app.py`: main demo app with 6 tabs
- `pages/1_methodology.py`: methodology / artifact browser page
- `pages/2_portfolio_summary.py`: Problem / Solution / Impact / Learning page
- `src/app/demo_data.py`: 앱에서 재사용하는 recommendation / cold-start / metrics loader
- `.streamlit/config.toml`: 로컬/클라우드 실행용 theme + server config

### 완료 기준
- [x] 주요 시나리오를 앱에서 직접 확인할 수 있음
- [x] 시연용 데모 흐름이 끊기지 않음

---

## Day 7. GitHub + Notion 정리
**예상 시간:** 4~5h  
**산출물:** README, GitHub 저장소 정리, Notion 포트폴리오 초안

### 작업 항목
- [x] `README.md` 작성 (개요, 데이터, 모델, 결과, 실행 방법)
- [x] 폴더 구조 정리
- [x] 핵심 시각화 3~4장 선별
- [x] Problem / Solution / Impact / Learning 초안 작성
- [x] Hybrid 모델 결과 및 베이스라인 대비 개선 포인트 정리
- [x] 왜 MovieLens를 사용했는지 설명 추가
- [x] 추천 시스템 A/B 테스트 설계를 포트폴리오 메시지와 연결
- [x] 한계점 + 다음 스텝 작성
- [x] 기술 스택 / 태그 / 글로우색상 정리

### 현재 준비된 지원 파일
- `README.md`
- `portfolio_summary.md`
- `ab_test_design.md`
- `artifacts/figures/day4_model_comparison_dashboard.png`
- `artifacts/figures/day4_model_tradeoff.png`
- `artifacts/figures/day5_cold_start_transition.png`

### 완료 기준
- [x] GitHub와 Notion에서 동일한 핵심 메시지가 전달됨
- [x] 면접/포트폴리오 설명에 바로 사용할 수 있는 문구가 준비됨

---

## 공통 체크리스트
- [x] GitHub README.md 정리
- [x] 핵심 시각화 스크린샷 3~4장 확보
- [x] `requirements.txt` 작성
- [ ] Streamlit 배포 URL 동작 확인
- [ ] Notion 포트폴리오 정리
- [x] 기술 스택 태그 정확히 기입
- [x] 글로우색상 설정
- [x] 비즈니스 맥락 1문단 작성
- [x] 한계점 + 다음 스텝 작성

## 메모 / 리스크
- Precision@10 목표값은 실제 실험 결과를 보고 확정한다.
- ALS / SVD++ / Neural CF는 설치 환경과 시간에 따라 난이도가 올라갈 수 있다.
- 기본 Python 3.13 환경에서는 `scikit-surprise`, `implicit` 호환성이 낮아 `requirements-optional.txt`로 분리했고, Day 2 실험은 `.uv311` (Python 3.11) 환경에서 실행했다.
- 태그 기반 특성은 MovieLens 메타데이터 가용성에 따라 범위를 조정할 수 있다.
- 오프라인 성능이 곧바로 온라인 성과를 의미하지 않으므로 A/B 테스트 설계 문서에서 그 간극을 설명해야 한다.
