# ============================================================
# 📌 Pipeline Overview (Flowchart Style) — Step 3: CLIP Embedding + Linear Probe
#
#  ROOT_DIRS (이미지 폴더들)        labels_zeroshot.csv
#             │                             │
#             ▼                             ▼
#   ┌───────────────────┐          ┌───────────────────────────┐
#   │ 1) 임베딩 로드/생성│          │ 2) 라벨 준비 (극단 분위수)│
#   │   - embeddings.npz │          │   - margin 상/하위 q%    │
#   │   - 없으면 새로 계산│          │   - pos=1 / neg=0        │
#   └──────────┬────────┘          └──────────────┬────────────┘
#              │ X, ids                          │ df_lab
#              └──────────────┬──────────────────┘
#                             ▼
#                 ┌───────────────────────────┐
#                 │ 3) X,y 라벨 정렬          │  <-- align_Xy(X, ids, df_lab)
#                 └─────────────┬─────────────┘
#                               │ X_all, y_all, folders_all
#                               ▼
#                 ┌───────────────────────────┐
#                 │ 4) 그룹 기반 Train/Val 분할│
#                 │   - GroupShuffleSplit      │
#                 │   - extract_group()        │
#                 └─────────────┬─────────────┘
#                        Xtr,ytr │ Xva,yva
#                               ▼
#                 ┌───────────────────────────┐
#                 │ 5) Logistic Regression 학습│
#                 │   - class_weight="balanced"│
#                 └─────────────┬─────────────┘
#                               │ clf
#                               ▼
#                 ┌───────────────────────────┐
#                 │ 6) 검증                   │
#                 │   - classification_report │
#                 │   - ROC-AUC               │
#                 │   - threshold 튜닝        │
#                 └─────────────┬─────────────┘
#                               │ th, recall, precision
#                               ▼
#                 ┌───────────────────────────┐
#                 │ 7) 저장                   │
#                 │   - linear_probe.npz      │
#                 │   - probe_meta.json       │
#                 └───────────────────────────┘
#
# Key Params:
#   MODEL_NAME, PRETRAINED, Q_EXTREME, VAL_RATIO,
#   TARGET_RECALL, RANDOM_SEED
#
# Outputs:
#   embeddings.npz (재사용)
#   linear_probe.npz (W, b, classes, meta)
#   probe_meta.json (threshold, counts, config)
# ============================================================

import os, json
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import open_clip
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

# ====== 경로 고정 ======
ROOT_DIRS = [
    r"C:/Users/user/Downloads/126.eye/0801/t/o/og/TS/TS/all_image_30/134_face_crop",
    r"C:/Users/user/Downloads/126.eye/0801/t/o/og/TS/TS/all_image_30/135_face_crop",
    r"C:/Users/user/Downloads/126.eye/0801/t/o/og/TS/TS/all_image_30/136_face_crop",
    r"C:/Users/user/Downloads/126.eye/01-1.data/Training/01.data/TS/001/T1/image_30_face_crop",
    r"C:/Users/user/Downloads/126.eye/01-1.data/Training/01.data/TS/002/T1/image_30_face_crop",
    r"C:/Users/user/Downloads/126.eye/01-1.data/Training/01.data/TS/003/T1/image_30_face_crop",
]
ZEROSHOT_CSV = r"C:\Users\user\Desktop\brainbuddy_AI\labels_zeroshot.csv"
EMB_PATH = "embeddings.npz"

# ====== 임베딩/모델 설정 ======
MODEL_NAME = "ViT-B-32"       # CLIP 백본
PRETRAINED = "openai"         # 사전학습 가중치
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_EXTS = (".png",".jpg",".jpeg",".bmp",".webp")
MAX_FRAMES = 30
BATCH_SIZE = 32

# ====== 학습 파라미터 ======
Q_EXTREME = 0.20      # margin 상/하위 20%씩 극단 샘플만 사용
VAL_RATIO = 0.2       # validation 비율
TARGET_RECALL = 0.90  # recall 우선 threshold 튜닝 목표
RANDOM_SEED = 42

# ===================== 유틸 함수들 =====================

def list_seq_folders(roots):
    """ROOT_DIRS 하위에서 이미지 포함된 모든 폴더 수집"""
    seqs = []
    for rd in roots:
        rd = os.path.abspath(rd)
        if not os.path.exists(rd): continue
        for cur, _, files in os.walk(rd):
            if any(f.lower().endswith(IMG_EXTS) for f in files):
                seqs.append(cur)
    return sorted(set(seqs))

def list_images(folder):
    """폴더 내 이미지 경로 리스트"""
    try:
        files = sorted(f for f in os.listdir(folder) if f.lower().endswith(IMG_EXTS))
        return [os.path.join(folder, f) for f in files]
    except:
        return []

def even_sample(items, k):
    """리스트에서 균등 간격으로 k개 샘플링"""
    if k<=0 or len(items)<=k: return items
    idx = np.linspace(0, len(items)-1, k, dtype=int)
    return [items[i] for i in idx]

def load_model():
    """CLIP 모델과 전처리기 로드"""
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED, device=DEVICE
    )
    model.eval()
    return model, preprocess

@torch.no_grad()
def embed_folder(model, preprocess, folder):
    """폴더 내 이미지들 → CLIP 임베딩 평균"""
    imgs = list_images(folder)
    if not imgs: return None
    sel = even_sample(imgs, MAX_FRAMES)
    xs = []
    for p in sel:
        try:
            xs.append(preprocess(Image.open(p).convert("RGB")))
        except:
            pass
    if not xs: return None
    X = torch.stack(xs,0).to(DEVICE)

    embs = []
    for i in range(0, len(X), BATCH_SIZE):
        v = model.encode_image(X[i:i+BATCH_SIZE])
        v = v / v.norm(dim=-1, keepdim=True)
        embs.append(v)
    v = torch.cat(embs,0).mean(0).cpu().numpy()  # 평균 pooling
    return v

def build_embeddings(folders, save_npz=EMB_PATH):
    """폴더 전체 임베딩 생성 후 npz 저장"""
    print("[Build] embeddings...")
    model, preprocess = load_model()
    X, ids = [], []
    for fd in tqdm(folders, desc="Embedding folders"):
        v = embed_folder(model, preprocess, fd)
        if v is None: continue
        X.append(v); ids.append(fd)
    X = np.stack(X,0)
    np.savez(save_npz, X=X, ids=np.array(ids, dtype=object),
             MODEL_NAME=MODEL_NAME, PRETRAINED=PRETRAINED)
    print(f"[Saved] {save_npz}  (N={len(ids)}, D={X.shape[1]})")
    return X, ids

def load_embeddings(npz_path=EMB_PATH):
    """기존 임베딩 로드, 없으면 새로 생성"""
    if not os.path.exists(npz_path):
        folders = list_seq_folders(ROOT_DIRS)
        return build_embeddings(folders, npz_path)
    data = np.load(npz_path, allow_pickle=True)
    X = data["X"]; ids = data["ids"]
    print(f"[Load] embeddings: N={len(ids)}, D={X.shape[1]})")
    return X, ids

# 그룹 추출: '.../<group>/<sequence_folder>' 구조에서 group을 상위 폴더명으로
def extract_group(folder_path: str) -> str:
    """폴더 경로에서 그룹명 추출 (사람/세션 단위 누수 방지용)"""
    p = os.path.normpath(folder_path)
    parts = p.split(os.sep)
    if len(parts) >= 2:
        return parts[-2]
    return parts[-1]

def split_train_val_grouped(X, y, folders, val_ratio=0.2, seed=42):
    """GroupShuffleSplit으로 그룹 단위 분할 → 데이터 누수 방지"""
    groups = np.array([extract_group(f) for f in folders])
    gss = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    tr_idx, va_idx = next(gss.split(X, y, groups))
    return (X[tr_idx], y[tr_idx], X[va_idx], y[va_idx],
            [folders[i] for i in tr_idx], [folders[i] for i in va_idx])

# ===================== 라벨 준비 =====================

def prepare_labels_extreme(df, q=Q_EXTREME):
    """
    극단 분위수 샘플만 사용해 라벨 생성:
    - margin 상위 (1-q) 분위수 이상 → pos(1)
    - margin 하위 q 분위수 이하  → neg(0)
    """
    need = ["folder","margin","p_focused"]
    df = df.dropna(subset=need).copy()

    low_thr  = float(np.quantile(df["margin"], q))
    high_thr = float(np.quantile(df["margin"], 1.0 - q))
    print(f"[Quantiles-extreme] low_thr={low_thr:.6f}, high_thr={high_thr:.6f}")

    neg_df = df[df["margin"] <= low_thr].copy()
    pos_df = df[df["margin"] >= high_thr].copy()

    neg_df["y"] = 0
    pos_df["y"] = 1

    # pos/neg 수를 동일하게 맞춤
    n = min(len(pos_df), len(neg_df))
    pos_df = pos_df.sample(n=n, random_state=RANDOM_SEED)
    neg_df = neg_df.sample(n=n, random_state=RANDOM_SEED)

    both = pd.concat([pos_df, neg_df], ignore_index=True).drop_duplicates(subset=["folder"])
    print(f"[Class sizes (extreme)] pos={len(pos_df)}, neg={len(neg_df)}")
    return both

def align_Xy(X, ids, df_lab):
    """임베딩 배열 X와 라벨 DataFrame을 align"""
    id2idx = {ids[i]: i for i in range(len(ids))}
    keep = [r.folder in id2idx for r in df_lab.itertuples()]
    df_lab = df_lab[keep].copy()
    idx = [id2idx[r.folder] for r in df_lab.itertuples()]
    X_sel = X[idx]
    y_sel = df_lab["y"].astype(int).values
    folders_sel = df_lab["folder"].tolist()
    return X_sel, y_sel, folders_sel

# ===================== 튜닝/학습 =====================

def tune_threshold_for_recall(y_true, p_prob, target_recall=0.90):
    """validation에서 recall ≥ 목표를 만족하는 threshold 탐색"""
    grid = np.linspace(0.1, 0.9, 161)
    best = None
    for th in grid:
        pred = (p_prob >= th).astype(int)
        tp = ((pred==1)&(y_true==1)).sum()
        fp = ((pred==1)&(y_true==0)).sum()
        fn = ((pred==0)&(y_true==1)).sum()
        recall = tp / max(tp+fn,1)
        precision = tp / max(tp+fp,1)
        score = (recall >= target_recall, precision, recall)
        if best is None or score > best[0]:
            best = (score, th, recall, precision)
    _, th, r, p = best
    return float(th), float(r), float(p)

# ===================== 메인 =====================

def main():
    # (0) 임베딩 로드/생성
    X, ids = load_embeddings(EMB_PATH)

    # (1) 라벨 준비 (극단 분위수 기반)
    if not os.path.exists(ZEROSHOT_CSV):
        raise FileNotFoundError(f"CSV not found: {ZEROSHOT_CSV}")
    df = pd.read_csv(ZEROSHOT_CSV)

    df_lab = prepare_labels_extreme(df, q=Q_EXTREME)
    X_all, y_all, folders_all = align_Xy(X, ids, df_lab)
    print(f"[Dataset aligned] N={len(y_all)}, pos={(y_all==1).sum()}, neg={(y_all==0).sum()}")

    # (2) 그룹 기반 train/val 분리 (데이터 누수 방지)
    Xtr, ytr, Xva, yva, ftr, fva = split_train_val_grouped(
        X_all, y_all, folders_all, val_ratio=VAL_RATIO, seed=RANDOM_SEED
    )
    print("[Groups] train unique groups:", len(set(map(extract_group, ftr))),
          "val unique groups:", len(set(map(extract_group, fva))))

    # (3) 로지스틱 회귀 학습
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(Xtr, ytr)

    # (4) 검증 성능 평가
    pred = clf.predict(Xva)
    prob = clf.predict_proba(Xva)[:,1]
    print("=== Validation Report ===")
    print(classification_report(yva, pred, digits=4))
    try:
        auc = roc_auc_score(yva, prob)
        print("ROC-AUC:", round(auc,4))
    except:
        pass

    # threshold 튜닝 (목표 recall 우선)
    th, r, p = tune_threshold_for_recall(yva, prob, target_recall=TARGET_RECALL)
    print(f"[Tuned] threshold={th:.3f}  (recall≈{r:.3f}, precision≈{p:.3f})")

    # (5) 모델 저장
    np.savez("linear_probe.npz",
             W=clf.coef_, b=clf.intercept_, classes_=clf.classes_,
             MODEL_NAME=MODEL_NAME, PRETRAINED=PRETRAINED)
    with open("probe_meta.json","w",encoding="utf-8") as f:
        json.dump({
            "threshold": float(th),
            "embedding_dim": int(X.shape[1]),
            "train_size": int(len(ytr)),
            "val_size": int(len(yva)),
            "pos_count": int((y_all==1).sum()),
            "neg_count": int((y_all==0).sum()),
            "model": MODEL_NAME, "pretrained": PRETRAINED
        }, f, ensure_ascii=False, indent=2)
    print("[Saved] linear_probe.npz, probe_meta.json")

if __name__ == "__main__":
    main()
