# ============================================================
# 📌 Pipeline Overview (Flowchart Style) — Step 4: Linear Probe Inference
#
#   embeddings.npz          linear_probe.npz         probe_meta.json(옵션)
#     (X, ids)                (W, b, meta)                   │
#         │                        │                         │
#         └──────────────┬─────────┴─────────────────────────┘
#                        ▼
#              ┌───────────────────────────┐
#              │ 1) 파일 존재 확인         │
#              │   - EMB_PATH, PROBE_PATH  │
#              └───────────────┬──────────┘
#                              │
#                              ▼
#              ┌───────────────────────────┐
#              │ 2) 로드                   │
#              │   - X, ids (임베딩)       │
#              │   - W, b (로지스틱 파라미터) │
#              │   - meta (선택적 정보)    │
#              └───────────────┬──────────┘
#                              │
#                              ▼
#              ┌───────────────────────────┐
#              │ 3) 예측                   │
#              │   - z = X·W + b           │
#              │   - p = sigmoid(z)        │
#              │   - pred = (p>=THRESHOLD) │
#              └───────────────┬──────────┘
#                              │
#                              ▼
#              ┌───────────────────────────┐
#              │ 4) 저장                   │
#              │   - labels_final.csv      │
#              │     • folder              │
#              │     • p_positive          │
#              │     • predicted_label     │
#              │     • threshold           │
#              │     • engine              │
#              │     • timestamp           │
#              └───────────────────────────┘
#
# Key Param:
#   THRESHOLD (최종 결정 임계값, default=0.700)
#
# Outputs:
#   labels_final.csv (최종 예측 결과)
# ============================================================


import os, json
import numpy as np
import pandas as pd
from datetime import datetime

# ====== 경로 ======
EMB_PATH = "embeddings.npz"        # 임베딩 npz
PROBE_PATH = "linear_probe.npz"    # 학습된 로지스틱 회귀 파라미터
META_PATH = "probe_meta.json"      # (옵션) 메타정보 JSON
OUT_CSV  = "labels_final.csv"      # 최종 결과 CSV

# ====== 최종 판정 임계값 ======
THRESHOLD = 0.700   # tuned threshold (예: recall 우선 조정값)

# ------------------------------------------------------------
# 유틸 함수
# ------------------------------------------------------------
def sigmoid(z):
    """로지스틱 함수 (z → [0,1] 확률)"""
    return 1.0 / (1.0 + np.exp(-z))

# ------------------------------------------------------------
# 메인 실행
# ------------------------------------------------------------
def main():
    # 1) 필수 파일 체크
    for p in [EMB_PATH, PROBE_PATH]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing file: {p}")

    # 2) 임베딩 로드
    emb = np.load(EMB_PATH, allow_pickle=True)
    if not {"X", "ids"}.issubset(set(emb.files)):
        raise KeyError(f"{EMB_PATH} must contain 'X' and 'ids' keys, found: {emb.files}")

    X   = emb["X"]      # shape: (N, D) → 폴더별 임베딩
    ids = emb["ids"]    # shape: (N,)   → 폴더 경로 문자열
    N, D = X.shape
    print(f"[Load] embeddings: N={N}, D={D}")

    # 3) Linear Probe 로드
    probe = np.load(PROBE_PATH, allow_pickle=True)
    if not {"W", "b"}.issubset(set(probe.files)):
        raise KeyError(f"{PROBE_PATH} must contain 'W' and 'b' keys, found: {probe.files}")

    W = probe["W"].reshape(1, -1)     # (1, D)
    b = probe["b"].reshape(1,)        # (1,)
    if W.shape[1] != D:
        raise ValueError(f"Dim mismatch: W has {W.shape[1]} dims but X has {D}")

    # (선택) 엔진 정보 로드 (없으면 unknown)
    try:
        model_name = probe["MODEL_NAME"].item()
        pretrained = probe["PRETRAINED"].item()
        engine = f"{model_name}/{pretrained}"
    except Exception:
        engine = "unknown"

    # (선택) meta 파일 로드 (threshold, dataset 크기 등 추가 정보)
    if os.path.exists(META_PATH):
        try:
            with open(META_PATH, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            meta = {}
    else:
        meta = {}

    print(f"[Meta] threshold={THRESHOLD:.3f}  engine={engine}")

    # 4) 예측 수행
    # 로짓 z = X·W + b → sigmoid → 확률
    z = X @ W.T + b            # (N, 1)
    p = sigmoid(z).reshape(-1) # (N,)
    pred = (p >= THRESHOLD).astype(int)  # 이진 라벨

    # 5) 결과 저장
    df = pd.DataFrame({
        "folder": ids,                 # 입력 폴더 경로
        "p_positive": p,               # 1(집중) 확률
        "predicted_label": pred,       # 1=집중, 0=비집중
        "threshold": THRESHOLD,        # 사용한 threshold 기록
        "engine": engine,              # 사용 모델 엔진
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    })
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"[Saved] {OUT_CSV} (rows={len(df)})")

# ------------------------------------------------------------
if __name__ == "__main__":
    main()
