import os
import pickle
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from tqdm import tqdm
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, recall_score, f1_score
import math  # Positional Encoding을 위한 math 추가
from torch.cuda.amp import autocast, GradScaler #Mixed Precision

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
        
        
        while len(frames) < 30:
            frames.append(frames[-1] if frames else torch.zeros(3, 224, 224))
        
        video = torch.stack(frames[:30])  # 정확히 30개만

        fusion_path = os.path.join(folder_path, "fusion_features.pkl")
        if os.path.exists(fusion_path):
            with open(fusion_path, 'rb') as f:
                fusion = torch.tensor(pickle.load(f), dtype=torch.float32)
        else:
            fusion = torch.zeros(5)

        return video, fusion, torch.tensor(label, dtype=torch.float32)

# ------------------ Model (CNN은 기존과 동일) ------------------
class CNNEncoder(nn.Module):
    def __init__(self, output_dim=1280):
        super().__init__()
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

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(B * T, -1)
        x = self.fc(x)
        return x.view(B, T, -1)

# ------------------ 새로운 Transformer 기반 모델 ------------------
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
    def __init__(self, cnn_feat_dim=1280, fusion_feat_dim=5, d_model=128, nhead=8, num_layers=3):
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

# ------------------ Training Functions (기존과 동일) ------------------
def train(model_cnn, model_top, loader, criterion, optimizer, device, scaler, accumulation_steps=4):
    model_cnn.train()
    model_top.train()
    total_loss = 0
    optimizer.zero_grad()

    for i, (videos, fusion, labels) in enumerate(tqdm(loader, desc="Train")):
        # non_blocking으로 GPU 전송 최적화
        videos = videos.to(device, non_blocking=True)
        fusion = fusion.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).unsqueeze(1)

        # Mixed Precision 적용 - autocast로 감싸기
        with autocast():
            features = model_cnn(videos)
            output = model_top(features, fusion)
            loss = criterion(output, labels)

        # 기존 loss.backward()를 scaler로 변경
        scaler.scale(loss).backward()
        total_loss += loss.item()

        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(loader):
            # 그래디언트 클리핑도 scaler와 함께 사용
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(model_cnn.parameters()) + list(model_top.parameters()), 
                max_norm=1.0
            )
            # optimizer.step()을 scaler로 변경
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

    return total_loss / len(loader)

def validate(model_cnn, model_top, loader, criterion, device):
    model_cnn.eval()
    model_top.eval()
    total_loss = 0

    with torch.no_grad():
        for videos, fusion, labels in tqdm(loader, desc="Validation"):
            # non_blocking 최적화
            videos = videos.to(device, non_blocking=True)
            fusion = fusion.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).unsqueeze(1)
            
            # Mixed Precision 적용
            with autocast():
                features = model_cnn(videos)
                outputs = model_top(features, fusion)
                loss = criterion(outputs, labels)
            total_loss += loss.item()

    return total_loss / len(loader)

def load_data(pkl_files):
    all_data = []
    for pkl_path in pkl_files:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
            all_data.extend(data)
    
    # 로딩 후 즉시 셔플링
    import random
    random.shuffle(all_data)
    return all_data


def check_batch_distribution(loader, num_batches=5):
    """배치별 클래스 분포 확인"""
    print("=" * 50)
    print("배치별 클래스 분포 확인")
    print("=" * 50)
    
    for i, (videos, fusion, labels) in enumerate(loader):
        if i >= num_batches:
            break
        
        class_0_count = (labels == 0).sum().item()
        class_1_count = (labels == 1).sum().item()
        total = len(labels)
        
        print(f"Batch {i+1}: Class 0: {class_0_count}/{total} ({class_0_count/total:.1%}) | Class 1: {class_1_count}/{total} ({class_1_count/total:.1%})")
    
    print("=" * 50)

def evaluate_and_save_confusion_matrix(model_cnn, model_top, loader, device, epoch):
    model_cnn.eval()
    model_top.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for videos, fusion, labels in loader:
            videos, fusion = videos.to(device), fusion.to(device)
            features = model_cnn(videos)
            outputs = model_top(features, fusion)
            preds = (torch.sigmoid(outputs) > 0.3).int().cpu().numpy()
            labels = labels.int().numpy()
            all_preds.extend(preds.flatten())
            all_labels.extend(labels.flatten())

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - Epoch {epoch+1}")
    plt.savefig(f"./log/confusion_matrix/train1/conf_matrix_epoch_{epoch+1}.png")
    plt.close()
    print(f"Confusion matrix saved: conf_matrix_epoch_{epoch+1}.png")

def evaluate_metrics(model_cnn, model_top, loader, device):
    model_cnn.eval()
    model_top.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for videos, fusion, labels in loader:
            videos, fusion = videos.to(device), fusion.to(device)
            features = model_cnn(videos)
            outputs = model_top(features, fusion)
            preds = (torch.sigmoid(outputs) > 0.3).int().cpu().numpy()
            labels = labels.int().numpy()
            all_preds.extend(preds.flatten())
            all_labels.extend(labels.flatten())

    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    return recall, f1

# ------------------ Main Function (기존과 동일, 모델 초기화 부분만 변경) ------------------
def main():
    torch.backends.cudnn.benchmark = True  # 성능 향상
    torch.cuda.empty_cache()  # 메모리 정리
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 디바이스: {device}")
    
    # GPU 메모리 사용량 확인
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        
    # 수정된 경로: 사용자가 제공한 Desktop 기반 경로 사용
    base_path = r"C:\Users\user\Desktop\brainbuddy_AI\preprocess2\pickle_labels"
    train_pkl_files = [
        f"{base_path}\\train\\20_01.pkl",
        f"{base_path}\\train\\20_03.pkl"
    ]
    val_pkl_files = [
        f"{base_path}\\valid\\20_01.pkl",
        f"{base_path}\\valid\\20_03.pkl"
    ]

    train_data_list = load_data(train_pkl_files)
    val_data_list = load_data(val_pkl_files)

    train_dataset = VideoFolderDataset(train_data_list)
    val_dataset = VideoFolderDataset(val_data_list)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=8, pin_memory=True)

    print("Training 데이터 배치 분포 확인:")
    check_batch_distribution(train_loader, num_batches=3)
    
    print("Validation 데이터 배치 분포 확인:")
    check_batch_distribution(val_loader, num_batches=3)

    # 모델 초기화 (Transformer 기반으로 변경)
    cnn = CNNEncoder().to(device)
    model = EngagementModel(d_model=128, nhead=8, num_layers=4).to(device)
    pos_weight = torch.tensor([1.2]).to(device)  # 소수 클래스에 더 높은 가중치
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(list(cnn.parameters()) + list(model.parameters()), lr=1e-5) 
    scaler = GradScaler()  # Mixed Precision을 위한 Scaler 생성
    
    best_val_loss = float('inf')
    best_model_path = "./log/best_model2.pt"
    checkpoint_path = "./log/last_checkpoint2.pt"
    log_history = []

    start_epoch = 0
    patience = 3
    patience_counter = 0

    """
    if os.path.exists(checkpoint_path):
        print(f"Resuming training from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        cnn.load_state_dict(ckpt['cnn_state_dict'])
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        print(f"Resumed from epoch {start_epoch} (best_val_loss={best_val_loss:.4f})")
    """

    num_epochs = 10
    for epoch in range(start_epoch, num_epochs):
        train_loss = train(cnn, model, train_loader, criterion, optimizer, device, scaler, accumulation_steps=4)
        val_loss = validate(cnn, model, val_loader, criterion, device)

        print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        recall, f1 = evaluate_metrics(cnn, model, val_loader, device)

        log_history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "recall": recall,
            "f1_score": f1
        })

        evaluate_and_save_confusion_matrix(cnn, model, val_loader, device, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'cnn_state_dict': cnn.state_dict(),
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss
            }, best_model_path)
            print(f"Best model saved at epoch {epoch+1} with val_loss {val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"Early stopping patience {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"==== Early stopping Triggered===")
                break

        torch.save({
            'cnn_state_dict': cnn.state_dict(),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'best_val_loss': best_val_loss
        }, checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch+1}")

    log_df = pd.DataFrame(log_history)
    log_df.to_csv("./log/train_log2.csv", index=False)
    print("Training log saved to train_log.csv")

    checkpoint = torch.load(best_model_path, map_location=device)
    cnn.load_state_dict(checkpoint['cnn_state_dict'])
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch']+1} (val_loss={checkpoint['val_loss']:.4f})")

if __name__ == '__main__':
    main()
