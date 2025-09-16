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

# ------------------ 개선된 데이터 증강 Dataset ------------------
class VideoFolderDataset(Dataset):
    def __init__(self, data_list, transform=None, is_training=True):
        self.data_list = []
        self.is_training = is_training
        
        # 개선된 데이터 증강 (훈련/검증 구분)
        if is_training:
            self.transform = transform or transforms.Compose([
                transforms.Resize((256, 256)),  # 더 큰 해상도
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # 랜덤 크롭
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
                transforms.RandomRotation(degrees=5),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            # 검증용은 기본 변환만
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

# ------------------ 개선된 CNN Encoder ------------------
class CNNEncoder(nn.Module):
    def __init__(self, output_dim=1280):
        super().__init__()
        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.features = mobilenet.features
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        
        # 더 큰 FC 레이어와 개선된 정규화
        self.fc = nn.Sequential(
            nn.Linear(1280 * 4 * 4, 2048),
            nn.BatchNorm1d(2048),  # BatchNorm 추가
            nn.ReLU(),
            nn.Dropout(0.4),  # 드롭아웃 증가
            nn.Linear(2048, output_dim),
            nn.BatchNorm1d(output_dim),
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

# ------------------ 개선된 Transformer 모델 ------------------
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

class EngagementModelV2(nn.Module):
    def __init__(self, cnn_feat_dim=1280, fusion_feat_dim=5, d_model=256, nhead=8, num_layers=4):  # ✅ 더 큰 모델
        super().__init__()
        
        # 입력 프로젝션: CNN 특징을 Transformer 차원으로 변환
        self.input_projection = nn.Sequential(
            nn.Linear(cnn_feat_dim, d_model),
            nn.LayerNorm(d_model),  # LayerNorm 추가
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # 개선된 Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.15,  # 드롭아웃 증가
            activation='gelu',  # ReLU → GELU
            batch_first=True,
            norm_first=True  # Pre-LN 구조
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 개선된 Pooling (Max + Average 조합)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        # 더 복잡한 최종 분류기
        self.fc = nn.Sequential(
            nn.Linear(d_model * 2 + fusion_feat_dim, 512),  # Max + Avg pooling
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )

    def forward(self, cnn_feats, fusion_feats):
        # 입력 프로젝션
        x = self.input_projection(cnn_feats)  # (B, T, d_model)
        
        # Positional Encoding 추가
        x = x.transpose(0, 1)  # (T, B, d_model)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # (B, T, d_model)
        
        # Transformer Encoder
        transformer_out = self.transformer_encoder(x)  # (B, T, d_model)
        
        # Max + Average Pooling 조합
        avg_pooled = self.global_avg_pool(transformer_out.transpose(1, 2)).squeeze(-1)  # (B, d_model)
        max_pooled = self.global_max_pool(transformer_out.transpose(1, 2)).squeeze(-1)  # (B, d_model)
        pooled = torch.cat([avg_pooled, max_pooled], dim=1)  # (B, d_model * 2)
        
        # Fusion features 결합
        combined = torch.cat([pooled, fusion_feats], dim=1)  # (B, d_model * 2 + 5)
        
        return self.fc(combined)

# ------------------ 기존 Training Functions (동일) ------------------
def train(model_cnn, model_top, loader, criterion, optimizer, device, scaler, accumulation_steps=4):
    model_cnn.train()
    model_top.train()
    total_loss = 0
    optimizer.zero_grad()

    for i, (videos, fusion, labels) in enumerate(tqdm(loader, desc="Train")):
        videos = videos.to(device, non_blocking=True)
        fusion = fusion.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).unsqueeze(1)

        with autocast():
            features = model_cnn(videos)
            output = model_top(features, fusion)
            loss = criterion(output, labels)

        scaler.scale(loss).backward()
        total_loss += loss.item()

        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(model_cnn.parameters()) + list(model_top.parameters()), 
                max_norm=1.0
            )
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
            videos = videos.to(device, non_blocking=True)
            fusion = fusion.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).unsqueeze(1)
            
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

def evaluate_and_save_confusion_matrix(model_cnn, model_top, loader, device, epoch, save_dir):
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
    plt.title(f"Confusion Matrix V2 - Epoch {epoch+1}")
    
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/conf_matrix_v2_epoch_{epoch+1}.png")
    plt.close()
    print(f"📊 Confusion matrix saved: conf_matrix_v2_epoch_{epoch+1}.png")

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

# ------------------ Main Function (Version 2) ------------------
def main():
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 디바이스: {device}")
    print(" **Version 2 - 개선된 모델 훈련 시작**")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    # Version 2 전용 경로 설정 (기존 모델 보존)
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

    # 개선된 데이터셋 (훈련/검증 구분)
    train_dataset = VideoFolderDataset(train_data_list, is_training=True)   # 데이터 증강 적용
    val_dataset = VideoFolderDataset(val_data_list, is_training=False)      # 기본 변환만

    train_loader = DataLoader(train_dataset, batch_size=3, shuffle=True, num_workers=6, pin_memory=True)  # 배치 크기 조정
    val_loader = DataLoader(val_dataset, batch_size=3, shuffle=False, num_workers=6, pin_memory=True)

    print("Training 데이터 배치 분포 확인:")
    check_batch_distribution(train_loader, num_batches=3)
    
    print("Validation 데이터 배치 분포 확인:")
    check_batch_distribution(val_loader, num_batches=3)

    # Version 2 개선된 모델 초기화
    cnn = CNNEncoder().to(device)
    model = EngagementModelV2(d_model=256, nhead=8, num_layers=4).to(device)  # 더 큰 모델
    
    # 개선된 Loss & Optimizer
    pos_weight = torch.tensor([1.5]).to(device)  # 가중치 증가
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # AdamW + 학습률 스케줄러
    optimizer = torch.optim.AdamW(
        list(cnn.parameters()) + list(model.parameters()), 
        lr=3e-6,           # 더 낮은 학습률
        weight_decay=1e-4,  # L2 정규화
        betas=(0.9, 0.999)
    )
    
    # 코사인 어닐링 스케줄러
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=15, eta_min=1e-7
    )
    
    scaler = GradScaler()
    
    # Version 2 전용 저장 경로
    os.makedirs("./log/v2", exist_ok=True)
    best_model_path = "./log/v2/best_model_v2.pt"
    checkpoint_path = "./log/v2/last_checkpoint_v2.pt"
    confusion_save_dir = "./log/v2/confusion_matrix"
    log_history = []

    start_epoch = 0
    patience = 5  # patience 증가
    patience_counter = 0
    best_val_loss = float('inf')

    # Resume 기능 (필요시 주석 해제)
    """
    if os.path.exists(checkpoint_path):
        print(f"Resuming training from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        cnn.load_state_dict(ckpt['cnn_state_dict'])
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        print(f"Resumed from epoch {start_epoch} (best_val_loss={best_val_loss:.4f})")
    """

    num_epochs = 15  # 에포크 증가
    
    print(f" **하이퍼파라미터 설정**")
    print(f"   - 모델 크기: d_model={model.input_projection[0].out_features}, layers={4}")
    print(f"   - 학습률: {3e-6}, Weight decay: {1e-4}")
    print(f"   - 배치 크기: {3}, 에포크: {num_epochs}")
    print(f"   - 데이터 증강: 활성화")
    print("=" * 60)
    
    for epoch in range(start_epoch, num_epochs):
        # 현재 학습률 출력
        current_lr = scheduler.get_last_lr()[0]
        print(f"[Epoch {epoch+1}/{num_epochs}] Learning Rate: {current_lr:.2e}")
        
        train_loss = train(cnn, model, train_loader, criterion, optimizer, device, scaler, accumulation_steps=4)
        val_loss = validate(cnn, model, val_loader, criterion, device)

        print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        recall, f1 = evaluate_metrics(cnn, model, val_loader, device)

        log_history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "recall": recall,
            "f1_score": f1,
            "learning_rate": current_lr
        })

        evaluate_and_save_confusion_matrix(cnn, model, val_loader, device, epoch, confusion_save_dir)

        # 베스트 모델 저장 (스케줄러 상태도 포함)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'cnn_state_dict': cnn.state_dict(),
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'hyperparameters': {
                    'd_model': 256,
                    'num_layers': 4,
                    'learning_rate': 3e-6,
                    'weight_decay': 1e-4
                }
            }, best_model_path)
            print(f"Best model V2 saved at epoch {epoch+1} with val_loss {val_loss:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"Early stopping patience {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"==== Early stopping Triggered ====")
                break

        # 체크포인트 저장
        torch.save({
            'cnn_state_dict': cnn.state_dict(),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch': epoch,
            'best_val_loss': best_val_loss
        }, checkpoint_path)
        
        # 스케줄러 스텝
        scheduler.step()
        print(f"Checkpoint saved at epoch {epoch+1}")

    # 로그 저장
    log_df = pd.DataFrame(log_history)
    log_df.to_csv("./log/v2/train_log_v2.csv", index=False)
    print("Training log V2 saved to ./log/v2/train_log_v2.csv")

    # 베스트 모델 로드
    checkpoint = torch.load(best_model_path, map_location=device)
    cnn.load_state_dict(checkpoint['cnn_state_dict'])
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best V2 model from epoch {checkpoint['epoch']+1} (val_loss={checkpoint['val_loss']:.4f})")
    print(" **Version 2 모델 훈련 완료!**")

if __name__ == '__main__':
    main()
