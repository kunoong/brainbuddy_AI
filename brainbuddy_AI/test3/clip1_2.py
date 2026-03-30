# ============================================================
# 📌 Pipeline Overview (Flowchart Style) — Step 2: QA & Confidence Split
#
#            labels_zeroshot.csv
#                     │
#                     ▼
#        ┌──────────────────────────┐
#        │ 1) 로드 & 컬럼 점검      │  <-- REQUIRED_COLS 확인
#        └─────────────┬────────────┘
#                      │ df
#                      ▼
#        ┌──────────────────────────┐
#        │ 2) 요약 통계/분포        │  <-- describe(), 라벨 카운트, margin 분포
#        └─────────────┬────────────┘
#                      │ df["margin"]
#                      ▼
#        ┌──────────────────────────┐
#        │ 3) 분위수 히스토그램     │  <-- quantile_hist(k=12)
#        └─────────────┬────────────┘
#                      │
#                      ▼
#        ┌──────────────────────────┐
#        │ 4) |margin| 분위수 분할  │  <-- split_by_quantiles(high_q, mid_q)
#        │    - high / mid / low    │
#        └─────────────┬────────────┘
#          hi, md, lo  │ thresholds(mid, high)
#                      ▼
#        ┌──────────────────────────┐
#        │ 5) CSV 저장              │  <-- labels_high/mid/low_conf.csv
#        └─────────────┬────────────┘
#                      │
#                      ▼
#        ┌──────────────────────────┐
#        │ 6) threshold 추천 (옵션) │  <-- sign(margin) 의사 GT 기반
#        │    - target_recall 만족  │  <-- suggest_threshold()
#        │    - threshold_suggestion.json 저장
#        └──────────────────────────┘
#
# Key Params:
#   CSV_PATH, HIGH_Q, MID_Q, TARGET_RECALL, SUGGEST_THRESHOLD
#
# Outputs:
#   labels_high_conf.csv, labels_mid_conf.csv, labels_low_conf.csv
#   threshold_suggestion.json (옵션: SUGGEST_THRESHOLD=True)
# ============================================================

import os
import json
import numpy as np
import pandas as pd

# ====== 경로 고정 ======
CSV_PATH = r"C:\Users\user\Desktop\brainbuddy_AI\labels_zeroshot.csv"

# ====== 파라미터 ======
# 분위수 기준 (데이터 분포 자동 적응)
HIGH_Q = 0.90    # |margin|의 상위 10% → high confidence
MID_Q  = 0.60    # |margin|의 상위 40% → mid confidence
TARGET_RECALL = 0.90           # threshold 추천 시 목표 recall
SUGGEST_THRESHOLD = True       # threshold 자동 추천 여부

# 분석에 필요한 최소 컬럼
REQUIRED_COLS = ["folder", "p_focused", "margin", "predicted_label"]

# ------------------------------------------------------------
# 유틸리티 함수
# ------------------------------------------------------------
def check_columns(df: pd.DataFrame):
    """CSV에 필수 컬럼이 모두 있는지 확인"""
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing}\nCSV 컬럼을 확인하세요.")

def summarize(df: pd.DataFrame):
    """기본 통계, 라벨별 개수, margin 분포 출력"""
    print("=== Basic Summary ===")
    print(df.describe(include="all"))

    print("\n=== Counts by predicted_label ===")
    if "predicted_label" in df.columns:
        print(df["predicted_label"].value_counts(dropna=False))

    print("\n=== Margin describe ===")
    print(df["margin"].describe())

def quantile_hist(series: pd.Series, k: int = 12):
    """|margin| 분포를 분위수(bin) 단위로 출력"""
    s = series.dropna()
    qs = np.linspace(0, 1, k + 1)
    edges = np.quantile(s, qs)
    edges = np.unique(edges)  # 중복 제거
    bins = pd.cut(s, bins=edges, include_lowest=True)
    vc = bins.value_counts().sort_index()
    return vc

def split_by_quantiles(df: pd.DataFrame, high_q: float, mid_q: float):
    """
    |margin| 값 기준으로 샘플을 high/mid/low로 나눔
    - high: 신뢰도 가장 높은 샘플
    - mid: 중간 신뢰도
    - low: 애매하거나 불확실
    """
    abs_m = df["margin"].abs()
    high_thr = float(np.quantile(abs_m, high_q))
    mid_thr  = float(np.quantile(abs_m, mid_q))
    print(f"\n[Auto thresholds] |margin| mid_thr={mid_thr:.6f}, high_thr={high_thr:.6f}")

    hi = df[abs_m >= high_thr].copy()
    md = df[(abs_m >= mid_thr) & (abs_m < high_thr)].copy()
    lo = df[abs_m < mid_thr].copy()
    return hi, md, lo, high_thr, mid_thr

def save_splits(hi: pd.DataFrame, md: pd.DataFrame, lo: pd.DataFrame):
    """분리된 high/mid/low 샘플 CSV 저장"""
    hi.to_csv("labels_high_conf.csv", index=False, encoding="utf-8-sig")
    md.to_csv("labels_mid_conf.csv", index=False, encoding="utf-8-sig")
    lo.to_csv("labels_low_conf.csv", index=False, encoding="utf-8-sig")
    print("\nSaved:")
    print(f" - labels_high_conf.csv  (rows={len(hi)})")
    print(f" - labels_mid_conf.csv   (rows={len(md)})")
    print(f" - labels_low_conf.csv   (rows={len(lo)})")

def suggest_threshold(df: pd.DataFrame, target_recall: float = 0.90):
    """
    p_focused의 threshold를 근사 추천.
    - 의사 GT(= sign(margin)) 사용
      margin > 0 → focused(1)
      margin < 0 → unfocused(0)
    - p_focused 임계값을 스윕하며 목표 recall을 만족하는 threshold 탐색
    """
    work = df.dropna(subset=["p_focused","margin"]).copy()
    if len(work) == 0:
        return None, None, None

    # margin 부호를 기준으로 의사 Ground Truth 생성
    y_true = (work["margin"] > 0).astype(int).values
    p = work["p_focused"].astype(float).values

    # 후보 threshold grid 생성
    grid = np.linspace(max(0.05, p.min()), min(0.95, p.max()), 181)
    best = None
    for th in grid:
        pred = (p >= th).astype(int)
        tp = ((pred == 1) & (y_true == 1)).sum()
        fp = ((pred == 1) & (y_true == 0)).sum()
        fn = ((pred == 0) & (y_true == 1)).sum()

        # 근사 precision / recall
        recall = tp / max(tp + fn, 1)
        precision = tp / max(tp + fp, 1)

        # 목표 recall을 만족하는 경우만 고려
        score = (recall >= target_recall, precision, recall)
        if best is None or score > best[0]:
            best = (score, th, recall, precision)

    if best is None:
        return None, None, None
    _, th, r, p_ = best
    return float(th), float(r), float(p_)

# ------------------------------------------------------------
# 메인 실행부
# ------------------------------------------------------------
def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV를 찾을 수 없습니다: {CSV_PATH}")

    # CSV 로드 및 기본 점검
    df = pd.read_csv(CSV_PATH)
    check_columns(df)
    summarize(df)

    # margin 분포 출력
    print("\n=== Margin histogram (quantile bins) ===")
    print(quantile_hist(df["margin"], k=12))

    # high/mid/low confidence 세트로 분할
    hi, md, lo, high_thr, mid_thr = split_by_quantiles(df, HIGH_Q, MID_Q)
    print(f"\nSplit counts → high(|m|>={high_thr:.6f}): {len(hi)}, "
          f"mid([{mid_thr:.6f},{high_thr:.6f})): {len(md)}, low(<{mid_thr:.6f}): {len(lo)}")
    save_splits(hi, md, lo)

    # threshold 추천 (옵션)
    if SUGGEST_THRESHOLD:
        th, r, p_ = suggest_threshold(df, target_recall=TARGET_RECALL)
        if th is None:
            print("\n[Suggest] threshold 추천에 실패했습니다.")
        else:
            print(f"\n[Suggest] threshold ≈ {th:.3f}  (approx recall={r:.3f}, precision={p_:.3f})")
            with open("threshold_suggestion.json", "w", encoding="utf-8") as f:
                json.dump({
                    "suggested_threshold": th,
                    "approx_recall": r,
                    "approx_precision": p_,
                    "auto_margins": {
                        "mid_quantile": MID_Q, "mid_thr": mid_thr,
                        "high_quantile": HIGH_Q, "high_thr": high_thr
                    }
                }, f, ensure_ascii=False, indent=2)
            print("Saved: threshold_suggestion.json")

if __name__ == "__main__":
    main()
