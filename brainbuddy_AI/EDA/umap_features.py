"""
UMAP Three‑Way Comparison — Analysis‑Only (Load .pt, No Training)
-----------------------------------------------------------------
Loads a trained MobileNet_v2 + LSTM model checkpoint (no late fusion),
extracts embeddings from your dataset, and visualizes three UMAP plots side‑by‑side:
  1) Pretrained CNN‑only (ImageNet MobileNet_v2)
  2) Trained CNN‑only (your CNNEncoder output averaged over frames)
  3) Trained CNN+LSTM last hidden state

You only need:
  - One or more pickle files listing [(folder_path, label), ...]
  - A checkpoint .pt that contains the trained weights (keys: 'cnn_state_dict' and 'model_state_dict')

Output:
  - ./log/analysis/umap_threeway.png

Requires:
  pip install torch torchvision tqdm scikit-learn umap-learn opencv-python pandas matplotlib
"""

import os
import pickle
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# =============================
# --------- Config ------------
# =============================
PICKLES = [
    # r"C:/KSEB/brainbuddy_AI/preprocess2/pickle_labels/valid/20_01.pkl",
    # r"C:/KSEB/brainbuddy_AI/preprocess2/pickle_labels/valid/20_03.pkl",
    r"C:/KSEB/brainbuddy_AI/preprocess2/pickle_labels/test/another_data.pkl"
]
CHECKPOINT_PATH = r"../log/train3_nolf/best_model/best_model_epoch_1.pt"  # <- 수정하세요
NUM_FRAMES = 30
BATCH_SIZE = 2
NUM_WORKERS = 8
OUT_FIG = "./analysis/umap_threeway_another.png"
N_NEIGHBORS = 30
MIN_DIST = 0.05
USE_PCA_DIM = 50   # UMAP 전에 차원 축소(속도/안정성)
RANDOM_STATE = 42

# =============================
# --------- Dataset -----------
# =============================
class VideoFolderDataset(Dataset):
    def __init__(self, data_list, transform=None, num_frames=30):
        self.num_frames = num_frames
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.data_list = []
        for folder_path, label in data_list:
            if os.path.isdir(folder_path):
                img_files = [f for f in os.listdir(folder_path)
                             if f.lower().endswith((".jpg", ".jpeg", ".png"))]
                if len(img_files) >= self.num_frames:
                    self.data_list.append((folder_path, label))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        folder_path, label = self.data_list[idx]
        img_files = sorted([f for f in os.listdir(folder_path)
                            if f.lower().endswith((".jpg", ".jpeg", ".png"))])[:self.num_frames]
        frames = []
        for f in img_files:
            img_path = os.path.join(folder_path, f)
            img = cv2.imread(img_path)
            if img is None:
                frames.append(torch.zeros(3, 224, 224))
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img)
            frames.append(self.transform(img_pil))
        video = torch.stack(frames)  # (T,3,224,224)
        return video, torch.tensor(label, dtype=torch.float32)

# =============================
# ---------- Models -----------
# =============================
class CNNEncoder(nn.Module):
    def __init__(self, output_dim=1280):
        super().__init__()
        if hasattr(models, 'MobileNet_V2_Weights'):
            mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        else:
            mobilenet = models.mobilenet_v2(pretrained=True)
        self.features = mobilenet.features
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))  # (1280,4,4) -> 20480
        self.fc = nn.Sequential(
            nn.Linear(1280 * 4 * 4, 2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2048, output_dim),
            nn.ReLU(),
        )

    def forward(self, x):  # x: (B,T,3,224,224)
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(B*T, -1)
        x = self.fc(x)
        return x.view(B, T, -1)  # (B,T,D)

class EngagementModelNoFusion(nn.Module):
    def __init__(self, cnn_feat_dim=1280, hidden_dim=128):
        super().__init__()
        self.lstm = nn.LSTM(input_size=cnn_feat_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, cnn_feats):
        _, (hn, _) = self.lstm(cnn_feats)
        x = hn.squeeze(0)
        return self.fc(x)

# =============================
# --------- Utilities ---------
# =============================

def load_data(pkl_files):
    all_data = []
    for pkl_path in pkl_files:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
            all_data.extend(data)
    return all_data


def load_checkpoint_safely(ckpt_path, device):
    """Load a checkpoint that (ideally) contains 'cnn_state_dict' and 'model_state_dict'.
    Falls back to common patterns if keys differ.
    Returns: dict with keys 'cnn_state_dict', 'model_state_dict' if available.
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    out = {"cnn_state_dict": None, "model_state_dict": None}
    if isinstance(ckpt, dict):
        # Standard case
        if 'cnn_state_dict' in ckpt and 'model_state_dict' in ckpt:
            out['cnn_state_dict'] = ckpt['cnn_state_dict']
            out['model_state_dict'] = ckpt['model_state_dict']
            return out
        # Sometimes saved as a single state_dict of a combined nn.Module
        for k in ['state_dict', 'model', 'model_state_dict']:
            if k in ckpt and isinstance(ckpt[k], dict):
                # Heuristic split not possible; return under model_state_dict
                out['model_state_dict'] = ckpt[k]
                return out
        # Or raw state_dict
        if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            out['model_state_dict'] = ckpt
            return out
    raise RuntimeError(f"Unsupported checkpoint format: {ckpt_path}")

# --------- Embedding collectors ---------
@torch.no_grad()
def build_pretrained_mobilenet_feature_extractor(device):
    if hasattr(models, 'MobileNet_V2_Weights'):
        mb = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    else:
        mb = models.mobilenet_v2(pretrained=True)
    feat_extractor = nn.Sequential(
        mb.features,
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten()
    ).eval().to(device)

    def extractor(batch_video):
        B, T, C, H, W = batch_video.shape
        x = batch_video.view(B*T, C, H, W).to(device)
        f = feat_extractor(x)                # (B*T,1280)
        return f.view(B, T, -1)              # (B,T,1280)
    return extractor

@torch.no_grad()
def collect_embeddings_pretrained_cnn(loader, device):
    extractor = build_pretrained_mobilenet_feature_extractor(device)
    X, y = [], []
    for videos, labels in tqdm(loader, desc="Embeddings: Pretrained CNN-only"):
        feats = extractor(videos)
        seq_vec = feats.mean(dim=1)
        X.append(seq_vec.cpu().numpy())
        y.append(labels.long().numpy())
    return np.vstack(X), np.concatenate(y)

@torch.no_grad()
def collect_embeddings_trained_cnn(cnn: nn.Module, loader, device):
    cnn.eval()
    X, y = [], []
    for videos, labels in tqdm(loader, desc="Embeddings: Trained CNN-only"):
        videos = videos.to(device)
        feats = cnn(videos)
        seq_vec = feats.mean(dim=1)
        X.append(seq_vec.cpu().numpy())
        y.append(labels.long().numpy())
    return np.vstack(X), np.concatenate(y)

@torch.no_grad()
def collect_embeddings_trained_lstm(cnn: nn.Module, top: nn.Module, loader, device):
    cnn.eval(); top.eval()
    X, y = [], []
    for videos, labels in tqdm(loader, desc="Embeddings: Trained CNN+LSTM"):
        videos = videos.to(device)
        feats = cnn(videos)
        lstm_out, (hn, cn) = top.lstm(feats)
        seq_vec = hn[-1]
        X.append(seq_vec.cpu().numpy())
        y.append(labels.long().numpy())
    return np.vstack(X), np.concatenate(y)

# --------- UMAP visualization ---------

def visualize_umap_three_sets(
    X_pre, y_pre, X_cnn, y_cnn, X_lstm, y_lstm,
    out_path,
    n_neighbors=30, min_dist=0.05, metric="euclidean",
    random_state=42, use_pca_dim=50
):
    try:
        import umap
    except Exception:
        raise RuntimeError("UMAP is not installed. `pip install umap-learn`")

    def preprocess(X):
        Xs = StandardScaler().fit_transform(X)
        if use_pca_dim is not None and Xs.shape[1] > use_pca_dim:
            Xs = PCA(n_components=use_pca_dim).fit_transform(Xs)
        return Xs

    Xp = preprocess(X_pre)
    Xc = preprocess(X_cnn)
    Xl = preprocess(X_lstm)

    reducer_args = dict(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist,
                        metric=metric, random_state=random_state)

    Zp = umap.UMAP(**reducer_args).fit_transform(Xp)
    Zc = umap.UMAP(**reducer_args).fit_transform(Xc)
    Zl = umap.UMAP(**reducer_args).fit_transform(Xl)

    plt.figure(figsize=(15, 4.8))

    plt.subplot(1, 3, 1)
    for l in np.unique(y_pre):
        idx = np.where(y_pre == l)[0]
        plt.scatter(Zp[idx, 0], Zp[idx, 1], s=8, alpha=0.6, label=str(l))
    plt.title("UMAP — Pretrained CNN-only")
    plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2")
    plt.legend(markerscale=3, fontsize=8)

    plt.subplot(1, 3, 2)
    for l in np.unique(y_cnn):
        idx = np.where(y_cnn == l)[0]
        plt.scatter(Zc[idx, 0], Zc[idx, 1], s=8, alpha=0.6, label=str(l))
    plt.title("UMAP — Trained CNN-only")
    plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2")
    plt.legend(markerscale=3, fontsize=8)

    plt.subplot(1, 3, 3)
    for l in np.unique(y_lstm):
        idx = np.where(y_lstm == l)[0]
        plt.scatter(Zl[idx, 0], Zl[idx, 1], s=8, alpha=0.6, label=str(l))
    plt.title("UMAP — Trained CNN+LSTM (last hidden)")
    plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2")
    plt.legend(markerscale=3, fontsize=8)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

# =============================
# ------------ main -----------
# =============================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Load dataset list
    data_list = []
    for p in PICKLES:
        with open(p, 'rb') as f:
            data_list.extend(pickle.load(f))

    dataset = VideoFolderDataset(data_list, num_frames=NUM_FRAMES)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)

    # 2) Build models and load checkpoint
    cnn = CNNEncoder().to(device)
    top = EngagementModelNoFusion().to(device)

    ck = load_checkpoint_safely(CHECKPOINT_PATH, device)
    if ck.get('cnn_state_dict'):
        cnn.load_state_dict(ck['cnn_state_dict'])
        print("Loaded cnn_state_dict from checkpoint")
    else:
        print("[WARN] cnn_state_dict not found in checkpoint — using randomly initialized CNNEncoder.")

    if ck.get('model_state_dict'):
        try:
            top.load_state_dict(ck['model_state_dict'])
            print("Loaded model_state_dict into EngagementModelNoFusion")
        except Exception:
            # If the checkpoint was a combined state_dict, try to load selectively
            missing, unexpected = top.load_state_dict(ck['model_state_dict'], strict=False)
            print(f"[WARN] Loaded model_state_dict (strict=False). Missing: {missing}, Unexpected: {unexpected}")
    else:
        print("[WARN] model_state_dict not found in checkpoint — LSTM will be randomly initialized.")

    # 3) Collect embeddings
    X_pre, y_pre = collect_embeddings_pretrained_cnn(loader, device)
    X_cnn, y_cnn = collect_embeddings_trained_cnn(cnn, loader, device)
    X_lstm, y_lstm = collect_embeddings_trained_lstm(cnn, top, loader, device)

    # 4) Visualize
    visualize_umap_three_sets(
        X_pre, y_pre, X_cnn, y_cnn, X_lstm, y_lstm,
        out_path=OUT_FIG,
        n_neighbors=N_NEIGHBORS,
        min_dist=MIN_DIST,
        metric="euclidean",
        random_state=RANDOM_STATE,
        use_pca_dim=USE_PCA_DIM,
    )
    print("✅ Saved:", os.path.abspath(OUT_FIG))


if __name__ == "__main__":
    main()
