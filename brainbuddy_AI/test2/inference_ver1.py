# inference_ver1.py
import os
import pickle
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import math  # Positional Encoding을 위해 추가

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    recall_score,
    f1_score
)

# ------------------ Dataset (기존과 동일) ------------------
class VideoFolderDataset(Dataset):
    def __init__(self, data_list, transform=None):
        self.data_list = []
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        for folder_path, label in data_list:
            if os.path.isdir(folder_path):
                img_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if len(img_files) >= 30:
                    self.data_list.append((folder_path, label))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        folder_path, label = self.data_list[idx]
        img_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])[:30]
        frames = []
        
        for f in img_files:
            img_path = os.path.join(folder_path, f)
            try:
                img_pil = Image.open(img_path).convert('RGB')
                frames.append(self.transform(img_pil))
            except Exception as e:
                print(f"이미지 로드 실패: {img_path}")
                continue
        
        # 30개 프레임 보장
        while len(frames) < 30:
            frames.append(frames[-1] if frames else torch.zeros(3, 224, 224))
        
        video = torch.stack(frames[:30])  # 정확히 30개만

        fusion_path = os.path.join(folder_path, "fusion_features.pkl")
        if os.path.exists(fusion_path):
            with open(fusion_path, 'rb') as f:
                fusion = torch.tensor(pickle.load(f), dtype=torch.float32)
        else:
            fusion = torch.zeros(5)

        # 라벨 처리 (문자열 → 숫자 변환)
        if isinstance(label, str):
            if label == '집중하지않음':
                label = 1
            else:
                label = 0  # 기본값

        return video, fusion, torch.tensor(label, dtype=torch.float32)

# ------------------ Model ------------------
class CNNEncoder(nn.Module):
    def __init__(self, output_dim=1280):
        super().__init__()
        # 훈련 모델과 동일한 weights 설정
        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.features = mobilenet.features
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Sequential(
            nn.Linear(1280 * 4 * 4, 2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2048, output_dim),
            nn.ReLU()
        )

    def forward(self, x):  # x: (B, 30, 3, 224, 224)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(B * T, -1)
        x = self.fc(x)
        return x.view(B, T, -1)

# Transformer 모델로 완전 교체
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(0)
        return x + self.pe[:seq_len, :].to(x.device)

class EngagementModel(nn.Module):
    def __init__(self, cnn_feat_dim=1280, fusion_feat_dim=5, d_model=128, nhead=8, num_layers=4):
        super().__init__()
        
        # 입력 프로젝션: CNN 특징을 Transformer 차원으로 변환
        self.input_projection = nn.Linear(cnn_feat_dim, d_model)
        
        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 시퀀스 집약을 위한 Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 최종 분류기
        self.fc = nn.Sequential(
            nn.Linear(d_model + fusion_feat_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

    def forward(self, cnn_feats, fusion_feats):
        # 입력 프로젝션
        x = self.input_projection(cnn_feats)  # (B, T, d_model)
        
        # Positional Encoding 추가 (시퀀스 순서 정보)
        x = x.transpose(0, 1)  # (T, B, d_model)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # (B, T, d_model)
        
        # Transformer Encoder
        transformer_out = self.transformer_encoder(x)  # (B, T, d_model)
        
        # Global Average Pooling으로 시퀀스 집약
        pooled = self.global_pool(transformer_out.transpose(1, 2)).squeeze(-1)  # (B, d_model)
        
        # Fusion features 결합
        combined = torch.cat([pooled, fusion_feats], dim=1)  # (B, d_model + 5)
        
        # 최종 출력
        return self.fc(combined)

# ------------------ Utils ------------------
def load_data(pkl_files):
    all_data = []
    for pkl_path in pkl_files:
        if not os.path.exists(pkl_path):
            print(f"파일을 찾을 수 없습니다: {pkl_path}")
            continue
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)  # [(folder_path, label), ...]
            all_data.extend(data)
            print(f"로드됨: {pkl_path} ({len(data)}개 샘플)")
    return all_data

# ------------------ Test only ------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 사용 디바이스: {device}")

    # 수정된 피클 파일 경로
    test_pkl_files = [
        "./preprocess2/pickle_labels/valid/20_02.pkl",
        "./preprocess2/pickle_labels/valid/20_04.pkl",
    ]
    
    # 수정된 모델 경로
    best_model_path = "./log/best_model2.pt"

    # 데이터 로드
    test_data_list = load_data(test_pkl_files)
    print(f"총 테스트 데이터: {len(test_data_list)}개")
    
    test_dataset = VideoFolderDataset(test_data_list)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=4, pin_memory=True)

    # Transformer 기반 모델 로드
    cnn = CNNEncoder().to(device)
    model = EngagementModel(d_model=128, nhead=8, num_layers=4).to(device)  # 훈련 모델과 동일한 하이퍼파라미터

    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Checkpoint not found: {best_model_path}")

    print(f"모델 로딩 중: {best_model_path}")
    ckpt = torch.load(best_model_path, map_location=device)
    cnn.load_state_dict(ckpt['cnn_state_dict'])
    model.load_state_dict(ckpt['model_state_dict'])
    cnn.eval()
    model.eval()
    print("모델 로딩 완료")

    # 추론 & 메트릭
    all_probs, all_preds, all_labels = [], [], []

    print("추론 시작...")
    with torch.no_grad():
        for videos, fusion, labels in tqdm(test_loader, desc="Test"):
            videos, fusion = videos.to(device, non_blocking=True), fusion.to(device, non_blocking=True)
            feats = cnn(videos)
            logits = model(feats, fusion)
            probs = torch.sigmoid(logits).squeeze(1).detach().cpu().numpy()
            preds = (probs >= 0.5).astype(np.int32)
            labels = labels.int().numpy()

            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    # 지표 계산
    acc = accuracy_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1  = f1_score(all_labels, all_preds, zero_division=0)
    cm  = confusion_matrix(all_labels, all_preds)

    print("\n" + "="*50)
    print("**테스트 결과**")
    print("="*50)
    print(f"Accuracy: {acc:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    
    # 클래스별 분포 출력
    unique, counts = np.unique(all_labels, return_counts=True)
    print(f"\n실제 라벨 분포: {dict(zip(unique, counts))}")
    unique, counts = np.unique(all_preds, return_counts=True)
    print(f"예측 라벨 분포: {dict(zip(unique, counts))}")

    # 혼동행렬 저장
    save_dir = "./log/test_validation"  # 명확한 구분
    os.makedirs(os.path.join(save_dir, "confusion_matrix"), exist_ok=True)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["집중함", "집중하지않음"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix - Validation Test")
    out_path = os.path.join(save_dir, "confusion_matrix", "conf_matrix_validation.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved: {out_path}")

if __name__ == "__main__":
    main()
