# data/

## 원칙
- 원본 데이터는 저장소 루트에 직접 두지 말고 `data/raw/` 아래에 둡니다.
- 대용량 원본 데이터는 Git에 커밋하지 않습니다.
- 모든 코드와 노트북은 **상대경로** 기준으로 데이터를 읽습니다.

## 권장 배치
- 공식 안내 페이지: `https://grouplens.org/datasets/movielens/100k/`
- 직접 다운로드 아카이브: `https://files.grouplens.org/datasets/movielens/ml-100k.zip`
- MovieLens 100K 공식 데이터셋 압축 해제 경로: `data/raw/ml-100k/`
- 예시 파일 경로:
  - `data/raw/ml-100k/u.data`
  - `data/raw/ml-100k/u.item`
  - `data/raw/ml-100k/u.user`
  - `data/raw/ml-100k/u.genre`

## 적재 전략
1. 원본 TSV/파이프 구분 파일을 `pandas`로 읽습니다.
2. 전처리된 테이블/피처는 `data/processed/`에 저장합니다.
3. 평가 결과 표는 `artifacts/metrics/`, 시각화는 `artifacts/figures/`에 저장합니다.
