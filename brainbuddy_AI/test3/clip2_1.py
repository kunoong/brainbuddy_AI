# clip2_1.py
# ============================================================
# 📌 Pipeline Overview (Flowchart Style)
#
#            labels_final.csv
#                   │
#                   ▼
#        ┌──────────────────────────┐
#        │ 1) Load & Sanity Check   │  <-- CSV 로드, 결측/경로 확인
#        └─────────────┬────────────┘
#                      │
#                      ▼
#        ┌──────────────────────────┐
#        │ 2) Group Key Extraction  │  <-- --group-preset / --group-mode / --group-regex
#        └─────────────┬────────────┘
#                      │ groups, y
#                      ▼
#        ┌──────────────────────────┐
#        │ 3) Group-Stratified Split│  <-- 누수 방지 + 각 split 클래스 보존
#        └─────────────┬────────────┘
#       train/val/test indices      │   (--strict-groups 실패 시 에러, --no-strict-groups면 폴백)
#                      ▼            │
#        ┌──────────────────────────┐
#        │ 4) Datasets & Dataloaders│  <-- seq_len, pad_strategy, transforms
#        └─────────────┬────────────┘
#                      │
#                      ▼
#        ┌──────────────────────────┐
#        │ 5) Model: CNN + LSTM     │  <-- resnet18/efficientnet_b0 + LSTM + head
#        └─────────────┬────────────┘
#                      │
#        ┌──────────────────────────┐
#        │ 6) Train (AMP optional)  │  <-- BCEWithLogits, Adam, early stopping by --monitor
#        └─────────────┬────────────┘
#                      │ best.ckpt
#                      ▼
#        ┌──────────────────────────┐
#        │ 7) Evaluate on Test      │  <-- Acc/Recall/F1/AUC, Report
#        └─────────────┬────────────┘
#                      │
#                      ▼
#        ┌──────────────────────────┐
#        │ 8) Save Artifacts        │  <-- confusion_matrix.png, roc_curve.png,
#        │                          │      metrics_log.csv, splits_indices.json
#        └──────────────────────────┘
#
# Key CLI flags:
# - Grouping: --group-preset / --group-mode / --group-regex / --group-depth
# - Safety:   --strict-groups / --no-strict-groups / --dry-run-split
# - Data:     --seq-len / --pad-strategy / --min-frames / --img-size
# - Train:    --monitor / --patience / --epochs / --lr / --batch-size / --workers
# ============================================================

# ------------------------------------------------------------
# CLIP 라벨 기반 CNN+LSTM 학습 (그룹 키 강화)
# - GPU/AMP(torch.amp), tqdm
# - 그룹 키 추출: --group-preset / --group-mode regex / pair_up/two_up/depth
# - 그룹 누수 방지 + 각 split 클래스 보존 시도
# - --strict-groups (기본 ON): 실패 시 에러 종료 (폴백 금지)
# - --dry-run-split: 스플릿만 점검하고 종료
# - 프레임 부족 자동 패딩
# ------------------------------------------------------------

import os, re, json, math, random, argparse, time, sys
from pathlib import Path
from contextlib import nullcontext

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from sklearn.metrics import (
    accuracy_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve
)
from sklearn.model_selection import StratifiedShuffleSplit

import matplotlib.pyplot as plt
from tqdm import tqdm


# =========================
# Utils (기본 유틸 함수)
# =========================
def set_seed(seed: int = 42):
    """랜덤 시드 고정 (재현성 보장)"""
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def exists(p: str) -> bool:
    """경로 존재 여부 체크"""
    try: return Path(p).exists()
    except: return False


# =========================
# 그룹 키 추출
# =========================
# 경로명에서 "사람 ID / 세션" 등을 그룹 키로 추출하기 위한 프리셋 정규식
PRESETS = {
    "auto_person": r"NIA22EYE_S\d+_(\d{3})_",             # 사람 ID만 추출
    "auto_person_session": r"NIA22EYE_(S\d+_\d{3})_T\d+", # 사람+세션
    "auto_dirid": r"/(\d{3})_face_crop/",                 # 상위 폴더 ID
}

def extract_group_from_path(folder: str, group_mode: str, depth: int, regex: str, group_preset: str):
    """
    그룹 키 추출 함수
    - group_preset이 있으면 우선 적용
    - 없으면 group_mode(pair_up/two_up/depth/regex)에 따라 적용
    """
    s = folder.replace("\\", "/")
    if group_preset and group_preset != "none":
        rgx = PRESETS.get(group_preset, None)
        if rgx:
            m = re.search(rgx, s)
            if m: return m.group(1)

    path = Path(folder)
    parts = path.parts

    if group_mode == "regex":
        m = re.search(regex, s)
        return m.group(1) if m else "UNKNOWN"
    if group_mode == "two_up":
        return parts[-3] if len(parts) >= 3 else parts[-1]
    if group_mode == "depth":
        cut = len(parts) - depth
        cut = max(1, cut)
        return "/".join(parts[max(0, cut-1):cut])

    # 기본값: pair_up (상위 2단계 폴더 사용)
    if len(parts) >= 4: return f"{parts[-4]}/{parts[-3]}"
    if len(parts) >= 3: return f"{parts[-3]}/{parts[-2]}"
    return parts[-1]


# =========================
# 그룹 분할 (클래스 보존 시도)
# =========================
def _split_counts(y_idx):
    """라벨 분포 통계 출력용"""
    y_sum = int(y_idx.sum())
    return {"n": len(y_idx), "pos": y_sum, "neg": len(y_idx)-y_sum}

def _fallback_sample_stratified(y, seed):
    """그룹 분할 실패 시 샘플 단위 stratified split로 폴백"""
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
    idx = np.arange(len(y))
    train_idx, temp_idx = next(sss1.split(idx, y))

    y_temp = y[temp_idx]
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    val_rel, test_rel = next(sss2.split(np.arange(len(y_temp)), y_temp))
    val_idx = temp_idx[val_rel]; test_idx = temp_idx[test_rel]
    return train_idx, val_idx, test_idx

def group_stratified_split_indices(groups, y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                                   seed=42, max_trials=3000, allow_fallback=False):
    """
    그룹 단위 stratified split
    - 그룹 누수 방지 (같은 그룹 train/val/test에 동시에 존재하지 않음)
    - 각 split에 0/1 클래스 모두 존재하도록 조건 탐색
    """
    rng = np.random.RandomState(seed)
    groups = np.asarray(groups); y = np.asarray(y)
    uniq = np.array(sorted(set(groups))); nG = len(uniq)

    if nG <= 2:
        if allow_fallback:
            return _fallback_sample_stratified(y, seed)
        raise RuntimeError(f"[SplitError] Too few groups (n_groups={nG}).")

    # 그룹 수 비율에 따라 train/val/test 그룹 수 결정
    n_train = max(1, int(round(train_ratio * nG)))
    n_val   = max(1, int(round(val_ratio   * nG)))
    if n_train + n_val >= nG:
        n_val = max(1, min(n_val, nG - 1))
        n_train = max(1, min(n_train, nG - n_val - 1))
    n_test  = max(1, nG - n_train - n_val)

    idx_all = np.arange(len(groups))

    # 조건 만족할 때까지 max_trials 반복
    for _ in range(max_trials):
        rng.shuffle(uniq)
        train_groups = set(uniq[:n_train])
        val_groups   = set(uniq[n_train:n_train+n_val])
        test_groups  = set(uniq[n_train+n_val:n_train+n_val+n_test])

        tr_idx = idx_all[np.isin(groups, list(train_groups))]
        va_idx = idx_all[np.isin(groups, list(val_groups))]
        te_idx = idx_all[np.isin(groups, list(test_groups))]

        cond = []
        for s in (tr_idx, va_idx, te_idx):
            if len(s)==0: cond.append(False); continue
            ys = y[s]
            cond.append(ys.sum()>0 and ys.sum()<len(ys))
        if all(cond):
            return tr_idx, va_idx, te_idx

    # 실패 시 폴백 여부 확인
    if allow_fallback:
        print("[Warn] Could not build group stratified splits; falling back to sample-level stratified split.")
        return _fallback_sample_stratified(y, seed)
    else:
        raise RuntimeError("[SplitError] Could not satisfy group+class constraints.")


# =========================
# 시각화 함수
# =========================
def save_confmat(figpath, y_true, y_pred, labels=(0,1)):
    """Confusion Matrix 그림 저장"""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(4,4), dpi=140)
    im = ax.imshow(cm, interpolation='nearest')
    ax.set_title("Confusion Matrix")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["0 (Unfocused)", "1 (Focused)"])
    ax.set_yticklabels(["0 (Unfocused)", "1 (Focused)"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(figpath); plt.close(fig)

def save_roc(figpath, y_true, y_prob):
    """ROC Curve 그림 저장"""
    try: auc = roc_auc_score(y_true, y_prob)
    except Exception: auc = None
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(4.5,4), dpi=140)
    ax.plot(fpr, tpr, label=f"AUC={auc:.4f}" if auc is not None else "AUC=N/A")
    ax.plot([0,1],[0,1], linestyle="--")
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right"); fig.tight_layout(); fig.savefig(figpath); plt.close(fig); return auc


# =========================
# Dataset (프레임 부족시 자동 패딩 포함)
# =========================
class SequenceDataset(Dataset):
    """
    폴더 단위 시퀀스 데이터셋
    - min_frames 미만이면 제외
    - seq_len 부족 시 pad_strategy(repeat_last/loop/blank 등)로 채움
    """
    def __init__(self, df, transform, seq_len=30, pad_strategy="repeat_last", min_frames=1):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.seq_len = seq_len
        self.pad_strategy = pad_strategy
        self.min_frames = min_frames

        # min_frames 미만 시퀀스 필터링
        keep_idx = []
        skipped = 0
        for i in range(len(self.df)):
            folder = str(self.df.iloc[i]["folder"])
            files = [f for f in os.listdir(folder) if f.lower().endswith((".png",".jpg",".jpeg"))]
            if len(files) >= self.min_frames:
                keep_idx.append(i)
            else:
                skipped += 1
        if skipped > 0:
            print(f"[Data] skipped {skipped} samples (< min_frames={self.min_frames})")
        self.df = self.df.iloc[keep_idx].reset_index(drop=True)
        self._pad_count = 0

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        folder = str(row["folder"])
        label = int(row["predicted_label"])

        files = sorted([f for f in os.listdir(folder) if f.lower().endswith((".png",".jpg",".jpeg"))])
        n = len(files)

        # 프레임 부족시 pad_strategy에 따라 채움
        if self.pad_strategy == "skip" and n < self.seq_len:
            raise RuntimeError(f"Insufficient frames with pad_strategy=skip: {folder} ({n}/{self.seq_len})")
        if n == 0:
            raise RuntimeError(f"No images in folder: {folder}")

        if n >= self.seq_len:
            files = files[:self.seq_len]
        else:
            deficit = self.seq_len - n
            if self.pad_strategy == "repeat_last":
                files = files + [files[-1]] * deficit
            elif self.pad_strategy == "loop":
                k = 0
                while len(files) < self.seq_len:
                    files.append(files[k % n]); k += 1
            elif self.pad_strategy == "blank":
                pass
            else:
                files = files + [files[-1]] * deficit
            self._pad_count += 1

        # 이미지 로드 및 변환
        frames = []
        for fn in files[:self.seq_len]:
            if fn is None and self.pad_strategy == "blank":
                img = Image.new("RGB", (224,224), (0,0,0))
            else:
                img = Image.open(os.path.join(folder, fn)).convert("RGB")
            img = self.transform(img)
            frames.append(img)
        if n < self.seq_len and self.pad_strategy == "blank":
            c, h, w = frames[0].shape
            deficit = self.seq_len - n
            for _ in range(deficit):
                frames.append(torch.zeros_like(frames[0]))
        x = torch.stack(frames, dim=0)  # (T,C,H,W)
        return x, label, folder

    @property
    def pad_count(self):
        return self._pad_count


# =========================
# Model 정의 (CNN + LSTM)
# =========================
class CNNEncoder(nn.Module):
    """프레임 단위 CNN feature extractor (ResNet18 or EfficientNet)"""
    def __init__(self, backbone="resnet18"):
        super().__init__()
        self.out_dim = 512
        if backbone == "resnet18":
            m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            self.encoder = nn.Sequential(*list(m.children())[:-1])  # (B,512,1,1)
            self.out_dim = 512
        elif backbone == "efficientnet_b0":
            m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            self.encoder = nn.Sequential(*list(m.features), nn.AdaptiveAvgPool2d((1,1)))
            self.out_dim = 1280
        else:
            raise ValueError("Unsupported backbone")

    def forward(self, x):  # (B,3,H,W)
        f = self.encoder(x)
        return f.view(f.size(0), -1)

class CNN_LSTM(nn.Module):
    """CNN feature + LSTM sequence model"""
    def __init__(self, backbone="resnet18", hidden=256, num_layers=2, bidirectional=True, dropout=0.3):
        super().__init__()
        self.cnn = CNNEncoder(backbone=backbone)
        self.lstm = nn.LSTM(
            input_size=self.cnn.out_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            dropout=0.0 if num_layers==1 else 0.2,
            bidirectional=bidirectional,
            batch_first=True,
        )
        d = 2 if bidirectional else 1
        self.head = nn.Sequential(
            nn.Linear(hidden*d, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
    def forward(self, x):  # (B,T,3,H,W)
        B,T,C,H,W = x.shape
        x = x.reshape(B*T, C, H, W)
        feats = self.cnn(x).view(B, T, -1)
        seq, _ = self.lstm(feats)
        pooled = seq.mean(dim=1)        # 평균 pooling
        return self.head(pooled).squeeze(1)


# =========================
# AMP & Device helpers
# =========================
class DummyScaler:
    """CPU fallback용 scaler (no-op)"""
    def scale(self, x): return x
    def step(self, opt): opt.step()
    def update(self): pass

def select_device(arg_device: str):
    """사용 디바이스 선택"""
    if arg_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    if arg_device == "cpu": return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_autocast_and_scaler(device):
    """AMP/GradScaler 반환"""
    if device.type == "cuda":
        return (lambda: torch.amp.autocast('cuda')), torch.amp.GradScaler('cuda')
    else:
        return (lambda: nullcontext()), DummyScaler()

def print_device_info(device):
    """GPU/CPU 환경 출력"""
    print("\n===== Device Info =====")
    print(f"Using device: {device}")
    if device.type == "cuda":
        idx = torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        cap = torch.cuda.get_device_capability(idx)
        total = torch.cuda.get_device_properties(idx).total_memory / (1024**3)
        print(f"GPU: {name} (capability {cap[0]}.{cap[1]})")
        print(f"VRAM total: {total:.2f} GB")
        print(f"CUDA: {torch.version.cuda}, cuDNN: {torch.backends.cudnn.version()}")
        torch.backends.cudnn.benchmark = True
        try: torch.set_float32_matmul_precision('high')
        except Exception: pass
    else:
        print("Running on CPU.")
    print("=======================\n")


# =========================
# Train / Eval
# =========================
def train_one_epoch(model, dl, optimizer, scaler, criterion, device, autocast_ctx):
    """한 epoch 학습 루프"""
    model.train()
    total = 0; running = 0.0
    pbar = tqdm(dl, desc="Train", ncols=100)
    for x, y, _ in pbar:
        x = x.to(device, non_blocking=True)
        y = y.float().to(device)

        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx():
            logit = model(x)
            loss = criterion(logit, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bs = x.size(0); total += bs; running += loss.item() * bs
        pbar.set_postfix(loss=f"{(running/max(1,total)):.4f}")
    return running / max(1,total)

@torch.no_grad()
def evaluate(model, dl, device, autocast_ctx, title="Val"):
    """검증/테스트 루프 (성능 지표 계산)"""
    model.eval()
    probs, ytrue, folders = [], [], []
    pbar = tqdm(dl, desc=title, ncols=100)
    for x, y, f in pbar:
        x = x.to(device, non_blocking=True)
        with autocast_ctx():
            logit = model(x)
            p = torch.sigmoid(logit).detach().cpu().numpy()
        probs.append(p); ytrue.append(y.numpy()); folders += list(f)
    probs = np.concatenate(probs)
    ytrue = np.concatenate(ytrue).astype(int)
    ypred = (probs >= 0.5).astype(int)

    acc = accuracy_score(ytrue, ypred)
    rec = recall_score(ytrue, ypred, zero_division=0)
    f1  = f1_score(ytrue, ypred, zero_division=0)
    try:
        auc = roc_auc_score(ytrue, probs)
    except Exception:
        auc = float("nan")
    report = classification_report(ytrue, ypred, digits=4, zero_division=0)
    return {"acc":acc, "recall":rec, "f1":f1, "auc":auc,
            "report":report, "y_true":ytrue, "y_prob":probs,
            "y_pred":ypred, "folders":folders}


# =========================
# Main (훈련 파이프라인)
# =========================
def main(args):
    set_seed(args.seed)
    device = select_device(args.device)
    autocast_ctx, scaler = get_autocast_and_scaler(device)
    print_device_info(device)

    # 1) CSV 로드
    df = pd.read_csv(args.csv)
    assert {"folder","predicted_label"}.issubset(df.columns), "CSV must have columns: folder, predicted_label"

    # 존재하지 않는 폴더 제거
    ok = df["folder"].apply(exists)
    if ok.sum() < len(df):
        df = df.loc[ok].reset_index(drop=True)
        print(f"[Warn] filtered non-existent folders. remain={len(df)}")

    # 2) 그룹 키 추출 + 라벨 준비
    groups = np.array([
        extract_group_from_path(
            str(p), args.group_mode, args.group_depth, args.group_regex, args.group_preset
        ) for p in df["folder"].tolist()
    ])
    y = df["predicted_label"].astype(int).to_numpy()

    # 3) 그룹 기반 stratified split
    try:
        tr_idx, va_idx, te_idx = group_stratified_split_indices(
            groups, y,
            train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
            seed=args.seed, max_trials=3000, allow_fallback=not args.strict_groups
        )
        fallback_used = False
    except RuntimeError as e:
        print(str(e))
        sys.exit(2)

    # split 통계 출력
    def stat(name, indices):
        ys = y[indices]
        return f"{name}: n={len(indices)}, pos={int(ys.sum())}, neg={int(len(ys)-ys.sum())}, groups={len(set(groups[indices]))}"
    print("[Split] ", stat("train", tr_idx), "|", stat("val", va_idx), "|", stat("test", te_idx))

    # dry-run 옵션 시 split만 저장하고 종료
    if args.dry_run_split:
        split_meta = {
            "args": vars(args),
            "train_indices": list(map(int, tr_idx)),
            "val_indices": list(map(int, va_idx)),
            "test_indices": list(map(int, te_idx)),
            "y_counts": {
                "train": _split_counts(y[tr_idx]),
                "val":   _split_counts(y[va_idx]),
                "test":  _split_counts(y[te_idx]),
            }
        }
        with open(args.out_splitmeta, "w", encoding="utf-8") as f:
            json.dump(split_meta, f, ensure_ascii=False, indent=2)
        print(f"[Saved] {args.out_splitmeta} (dry run).")
        return

    # 4) 이미지 변환 정의 (train vs eval)
    train_tfms = transforms.Compose([
        transforms.Resize(args.img_size+32),
        transforms.CenterCrop(args.img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.1,0.1,0.1,0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    eval_tfms = transforms.Compose([
        transforms.Resize(args.img_size+32),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    # 5) 데이터셋/로더 생성
    train_df = df.iloc[tr_idx].reset_index(drop=True)
    val_df   = df.iloc[va_idx].reset_index(drop=True)
    test_df  = df.iloc[te_idx].reset_index(drop=True)

    train_ds = SequenceDataset(train_df, transform=train_tfms, seq_len=args.seq_len,
                               pad_strategy=args.pad_strategy, min_frames=args.min_frames)
    val_ds   = SequenceDataset(val_df, transform=eval_tfms,  seq_len=args.seq_len,
                               pad_strategy=args.pad_strategy, min_frames=args.min_frames)
    test_ds  = SequenceDataset(test_df, transform=eval_tfms,  seq_len=args.seq_len,
                               pad_strategy=args.pad_strategy, min_frames=args.min_frames)

    pin = (device.type=="cuda")
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers, pin_memory=pin,
                          prefetch_factor=(2 if args.workers>0 else None),
                          persistent_workers=(True if args.workers>0 else False))
    val_dl   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.workers, pin_memory=pin,
                          prefetch_factor=(2 if args.workers>0 else None),
                          persistent_workers=(True if args.workers>0 else False))
    test_dl  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.workers, pin_memory=pin,
                          prefetch_factor=(2 if args.workers>0 else None),
                          persistent_workers=(True if args.workers>0 else False))

    print(f"[Pad] train padded: {train_ds.pad_count} | val padded: {val_ds.pad_count} | test padded: {test_ds.pad_count}")

    # 6) 모델/손실/옵티마 정의
    model = CNN_LSTM(backbone=args.backbone, hidden=args.hidden,
                     num_layers=args.num_layers, bidirectional=not args.unidirectional,
                     dropout=args.dropout).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 7) 학습 루프 (early stopping 포함)
    best_score = -np.inf; best_state = None; wait=0; log_rows=[]
    for epoch in range(1, args.epochs+1):
        t0=time.time()
        train_loss = train_one_epoch(model, train_dl, optimizer, scaler, criterion, device, autocast_ctx)
        val_res = evaluate(model, val_dl, device, autocast_ctx, title="Val")
        dt = time.time()-t0

        monitor = val_res[args.monitor] if not math.isnan(val_res.get(args.monitor, float("nan"))) else -np.inf
        print(f"\n=== Epoch {epoch} ({dt:.1f}s) ===")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val -> Acc {val_res['acc']:.4f} | Recall {val_res['recall']:.4f} | F1 {val_res['f1']:.4f} | AUC {val_res['auc']:.4f}")
        print(val_res["report"])

        log_rows.append({"epoch":epoch,"train_loss":train_loss,"val_acc":val_res['acc'],
                         "val_recall":val_res['recall'],"val_f1":val_res['f1'],
                         "val_auc":val_res['auc'],"epoch_time_sec":dt})

        # best 갱신 시 체크포인트 저장
        if monitor>best_score:
            best_score=monitor; wait=0
            best_state={"epoch":epoch,"model":model.state_dict(),"optimizer":optimizer.state_dict(),
                        "monitor":args.monitor,"best_score":best_score,"args":vars(args)}
            torch.save(best_state, args.out_ckpt)
            print(f"[Save] {args.out_ckpt} (best {args.monitor}={best_score:.4f})")
        else:
            wait+=1; print(f"[EarlyStopping] wait={wait}/{args.patience}")
            if wait>=args.patience:
                print("[EarlyStopping] patience reached. stop training."); break

    pd.DataFrame(log_rows).to_csv(args.out_log, index=False, encoding="utf-8-sig")
    print(f"[Saved] {args.out_log}")

    # 8) 테스트
    if best_state is not None:
        model.load_state_dict(best_state["model"])
    test_res = evaluate(model, test_dl, device, autocast_ctx, title="Test")
    print("\n[Test]")
    print(f"Acc {test_res['acc']:.4f} | Recall {test_res['recall']:.4f} | F1 {test_res['f1']:.4f} | AUC {test_res['auc']:.4f}")
    print(test_res["report"])

    # 9) 혼동행렬/ROC 저장 + split 메타 저장
    save_confmat(args.out_confmat, test_res["y_true"], test_res["y_pred"])
    auc = save_roc(args.out_roc, test_res["y_true"], test_res["y_prob"])
    print(f"[Saved] {args.out_confmat}, {args.out_roc} (AUC={auc})")

    split_meta = {
        "args": vars(args),
        "train_indices": list(map(int, tr_idx)),
        "val_indices": list(map(int, va_idx)),
        "test_indices": list(map(int, te_idx)),
        "y_counts": {
            "train": _split_counts(y[tr_idx]),
            "val":   _split_counts(y[va_idx]),
            "test":  _split_counts(y[te_idx]),
        }
    }
    with open(args.out_splitmeta, "w", encoding="utf-8") as f:
        json.dump(split_meta, f, ensure_ascii=False, indent=2)
    print(f"[Saved] {args.out_splitmeta}")


# =========================
# argparse (CLI 인자 정의)
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="labels_final.csv")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--seq-len", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--unidirectional", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18","efficientnet_b0"])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto","cuda","cpu"])
    parser.add_argument("--monitor", type=str, default="f1", choices=["f1","recall","acc","auc"])
    parser.add_argument("--patience", type=int, default=5)

    # 그룹 설정
    parser.add_argument("--group-preset", type=str, default="auto_person",
                        choices=["auto_person","auto_person_session","auto_dirid","none"],
                        help="자주 쓰는 경로 패턴 프리셋")
    parser.add_argument("--group-mode", type=str, default="regex", choices=["regex","pair_up","two_up","depth"],
                        help="프리셋 미사용시 모드 선택")
    parser.add_argument("--group-depth", type=int, default=3)
    parser.add_argument("--group-regex", type=str, default=r"NIA22EYE_S\d+_(\d{3})_",
                        help="group-mode=regex 에서 사용할 패턴")
    parser.add_argument("--strict-groups", dest="strict_groups", action="store_true", default=True,
                        help="그룹 분할 실패 시 에러로 종료 (폴백 금지)")
    parser.add_argument("--no-strict-groups", dest="strict_groups", action="store_false")

    # 패딩
    parser.add_argument("--pad-strategy", type=str, default="repeat_last", choices=["repeat_last","loop","blank","skip"])
    parser.add_argument("--min-frames", type=int, default=1)

    # 출력 파일들
    parser.add_argument("--out-ckpt", type=str, default="the_best.pth")
    parser.add_argument("--out-log", type=str, default="metrics_log.csv")
    parser.add_argument("--out-confmat", type=str, default="confusion_matrix.png")
    parser.add_argument("--out-roc", type=str, default="roc_curve.png")
    parser.add_argument("--out-splitmeta", type=str, default="splits_indices.json")

    # 훈련 전 점검만 수행하는 옵션
    parser.add_argument("--dry-run-split", action="store_true", help="스플릿만 계산하고 저장 후 종료")

    args = parser.parse_args()
    main(args)
