# ver1+ver2 ensemble 적용
import os
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from tqdm import tqdm
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, recall_score, f1_score, accuracy_score
import math
import numpy as np
from torch.cuda.amp import autocast, GradScaler
import time
import torch.nn.functional as F

# ------------------ 최적화된 Dataset ------------------
class VideoFolderDataset(Dataset):
    def __init__(self, data_list, transform=None, is_training=True):
        self.data_list = []
        self.is_training = is_training
        
        # 최소한의 변환으로 속도 최대화
        if is_training:
            self.transform = transform or transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.2),  # 확률 낮춤
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
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
            except Exception:
                continue
        
        while len(frames) < 30:
            frames.append(frames[-1] if frames else torch.zeros(3, 224, 224))
        
        video = torch.stack(frames[:30])

        fusion_path = os.path.join(folder_path, "fusion_features.pkl")
        if os.path.exists(fusion_path):
            with open(fusion_path, 'rb') as f:
                fusion = torch.tensor(pickle.load(f), dtype=torch.float32)
        else:
            fusion = torch.zeros(5)

        return video, fusion, torch.tensor(label, dtype=torch.float32)

# ------------------ 최적화된 CNNEncoder ------------------
class CNNEncoderV1(nn.Module):
    def __init__(self, output_dim=1280):
        super().__init__()
        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.features = mobilenet.features
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Sequential(
            nn.Linear(1280 * 4 * 4, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(2048, output_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(B * T, -1)
        x = self.fc(x)
        return x.view(B, T, -1)

class CNNEncoderV2(nn.Module):
    def __init__(self, output_dim=1280):
        super().__init__()
        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.features = mobilenet.features
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Sequential(
            nn.Linear(1280 * 4 * 4, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(2048, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(B * T, -1)
        x = self.fc(x)
        return x.view(B, T, -1)

# ------------------ Positional Encoding ------------------
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

# ------------------ 최적화된 Version 1 Transformer ------------------
class EngagementModelV1(nn.Module):
    def __init__(self, cnn_feat_dim=1280, fusion_feat_dim=5, d_model=128, nhead=8, num_layers=3):
        super().__init__()
        
        self.input_projection = nn.Linear(cnn_feat_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(d_model + fusion_feat_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

    def forward(self, cnn_feats, fusion_feats):
        x = self.input_projection(cnn_feats)
        x = x.transpose(0, 1)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)
        transformer_out = self.transformer_encoder(x)
        pooled = self.global_pool(transformer_out.transpose(1, 2)).squeeze(-1)
        combined = torch.cat([pooled, fusion_feats], dim=1)
        return self.fc(combined)

# ------------------ 최적화된 Version 2 Transformer ------------------
class EngagementModelV2(nn.Module):
    def __init__(self, cnn_feat_dim=1280, fusion_feat_dim=5, d_model=256, nhead=8, num_layers=4):
        super().__init__()
        
        self.input_projection = nn.Sequential(
            nn.Linear(cnn_feat_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.15,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(d_model * 2 + fusion_feat_dim, 512),
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
        x = self.input_projection(cnn_feats)
        x = x.transpose(0, 1)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)
        transformer_out = self.transformer_encoder(x)
        
        avg_pooled = self.global_avg_pool(transformer_out.transpose(1, 2)).squeeze(-1)
        max_pooled = self.global_max_pool(transformer_out.transpose(1, 2)).squeeze(-1)
        pooled = torch.cat([avg_pooled, max_pooled], dim=1)
        
        combined = torch.cat([pooled, fusion_feats], dim=1)
        return self.fc(combined)

# ------------------ 속도 최적화된 앙상블 모델 ------------------
class TransformerEnsembleModel(nn.Module):
    def __init__(self, cnn_v1, model_v1, cnn_v2, model_v2, ensemble_method='weighted'):
        super().__init__()
        self.cnn_v1 = cnn_v1
        self.model_v1 = model_v1
        self.cnn_v2 = cnn_v2
        self.model_v2 = model_v2
        self.ensemble_method = ensemble_method
        
        if ensemble_method == 'weighted':
            self.register_buffer('weights', torch.tensor([0.3, 0.7]))
        elif ensemble_method == 'learned':
            self.ensemble_weights = nn.Parameter(torch.tensor([0.3, 0.7]))
            self.ensemble_fc = nn.Sequential(
                nn.Linear(2, 16),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(16, 1)
            )

    def forward(self, videos, fusion_feats):
        # Mixed Precision과 함께 병렬 처리
        feats_v1 = self.cnn_v1(videos)
        feats_v2 = self.cnn_v2(videos)
        
        logits_v1 = self.model_v1(feats_v1, fusion_feats)
        logits_v2 = self.model_v2(feats_v2, fusion_feats)
        
        if self.ensemble_method == 'weighted':
            return self.weights[0] * logits_v1 + self.weights[1] * logits_v2
        elif self.ensemble_method == 'learned':
            prob_v1 = torch.sigmoid(logits_v1)
            prob_v2 = torch.sigmoid(logits_v2)
            normalized_weights = torch.softmax(self.ensemble_weights, dim=0)
            weighted_v1 = prob_v1 * normalized_weights[0]
            weighted_v2 = prob_v2 * normalized_weights[1]
            combined_input = torch.cat([weighted_v1, weighted_v2], dim=1)
            return self.ensemble_fc(combined_input)

# ------------------ 고속 훈련 함수 (Mixed Precision + Gradient Accumulation) ------------------
def train_ensemble_speed(ensemble_model, loader, criterion, optimizer, device, scaler, accumulation_steps=4):
    ensemble_model.train()
    # 사전훈련 모델들은 eval 모드 유지
    ensemble_model.cnn_v1.eval()
    ensemble_model.model_v1.eval()
    ensemble_model.cnn_v2.eval()
    ensemble_model.model_v2.eval()
    
    total_loss = 0
    batch_count = 0
    optimizer.zero_grad()

    for i, (videos, fusion, labels) in enumerate(tqdm(loader, desc="High-Speed Ensemble Train")):
        videos = videos.to(device, non_blocking=True)
        fusion = fusion.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).unsqueeze(1)

        # Mixed Precision 적용
        with autocast():
            output = ensemble_model(videos, fusion)
            loss = criterion(output, labels) / accumulation_steps  # gradient accumulation용 스케일링

        scaler.scale(loss).backward()
        
        # Gradient Accumulation으로 effective batch size 증가
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(loader):
            scaler.unscale_(optimizer)
            # Gradient clipping (선택사항)
            if ensemble_model.ensemble_method == 'learned':
                torch.nn.utils.clip_grad_norm_(
                    list(ensemble_model.ensemble_weights) + list(ensemble_model.ensemble_fc.parameters()), 
                    max_norm=1.0
                )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
        batch_count += 1

    return total_loss / batch_count

def validate_ensemble_speed(ensemble_model, loader, criterion, device, max_batches=None):
    ensemble_model.eval()
    total_loss = 0
    batch_count = 0

    with torch.no_grad():
        for i, (videos, fusion, labels) in enumerate(tqdm(loader, desc="High-Speed Validation")):
            if max_batches and i >= max_batches:
                break
                
            videos = videos.to(device, non_blocking=True)
            fusion = fusion.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).unsqueeze(1)
            
            # Mixed Precision 적용
            with autocast():
                outputs = ensemble_model(videos, fusion)
                loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            batch_count += 1

    return total_loss / batch_count

def load_data(pkl_files):
    all_data = []
    for pkl_path in pkl_files:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
            all_data.extend(data)
    
    import random
    random.shuffle(all_data)
    return all_data

def evaluate_ensemble_speed(ensemble_model, loader, device, threshold=0.7, max_batches=200):
    ensemble_model.eval()
    
    all_preds = []
    all_labels = []
    batch_count = 0
    
    with torch.no_grad():
        for videos, fusion, labels in tqdm(loader, desc="Fast Evaluation"):
            if batch_count >= max_batches:
                break
                
            videos, fusion = videos.to(device, non_blocking=True), fusion.to(device, non_blocking=True)
            
            with autocast():
                outputs = ensemble_model(videos, fusion)
                probs = torch.sigmoid(outputs).cpu().numpy()
            
            # 후처리 보정 로직 (이동평균 스무딩)
            p = probs.flatten()
            k = 3
            kernel = np.ones(k)/k
            p_smooth = np.convolve(p, kernel, mode='same')
            preds = (p_smooth > threshold).astype(int)

            
            all_preds.extend(preds.flatten())
            all_labels.extend(labels.int().numpy().flatten())
            batch_count += 1
    
    if len(all_labels) > 0:
        accuracy = accuracy_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        return accuracy, recall, f1
    else:
        return 0, 0, 0

# ------------------ 메인 함수 (속도 최적화) ------------------
def main():
    # 최대 GPU 최적화 설정
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.cuda.empty_cache()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 디바이스: {device}")
    print(" **속도 최적화된 전체 데이터 앙상블 훈련**")
    print("="*70)
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        # GPU 메모리 사용률 최대화
        torch.cuda.set_per_process_memory_fraction(0.95)
    
    ensemble_dir = "./log/ensemble"
    os.makedirs(ensemble_dir, exist_ok=True)
    
    # 전체 데이터 사용
    base_path = r"C:\Users\user\Desktop\brainbuddy_AI\preprocess2\pickle_labels"
    
    train_pkl_files = [
        f"{base_path}\\train\\20_01.pkl",
        f"{base_path}\\train\\20_03.pkl"  # 전체 훈련 데이터 사용
    ]
    val_pkl_files = [
        f"{base_path}\\valid\\20_01.pkl",
        f"{base_path}\\valid\\20_03.pkl"   # 전체 검증 데이터 사용
    ]

    train_data_list = load_data(train_pkl_files)
    val_data_list = load_data(val_pkl_files)

    # 전체 데이터 사용 (샘플링 없음)
    train_dataset = VideoFolderDataset(train_data_list, is_training=True)
    val_dataset = VideoFolderDataset(val_data_list, is_training=False)

    # 큰 배치 크기 + 최적화 설정
    batch_size = 8  # GPU 메모리가 허용하는 최대 크기
    accumulation_steps = 4  # effective batch size = 8 * 4 = 32
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=8,  # 워커 수 증가
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,  # 미리 로딩 수 증가
        drop_last=True  # 마지막 배치 드롭으로 속도 향상
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        drop_last=False
    )

    print(f"훈련 데이터: {len(train_data_list):,}개")
    print(f"검증 데이터: {len(val_data_list):,}개")
    print(f"훈련 배치: {len(train_loader):,}개 (배치크기: {batch_size}, Accumulation: {accumulation_steps})")
    print(f"Effective Batch Size: {batch_size * accumulation_steps}")

    # 사전 훈련된 모델들 로드
    print("\n Transformer 모델들 로드 중...")
    
    # Version 1 모델 로드
    cnn_v1 = CNNEncoderV1().to(device)
    model_v1 = EngagementModelV1(d_model=128, nhead=8, num_layers=3).to(device)
    
    try:
        v1_checkpoint = torch.load("./log/best_model2.pt", map_location=device)
        cnn_v1.load_state_dict(v1_checkpoint['cnn_state_dict'])
        model_v1.load_state_dict(v1_checkpoint['model_state_dict'])
        print("Version 1 Transformer 로드 완료")
    except Exception as e:
        print(f"Version 1 모델 로드 실패: {e}")
        return
    
    # Version 2 모델 로드
    cnn_v2 = CNNEncoderV2().to(device)
    model_v2 = EngagementModelV2(d_model=256, nhead=8, num_layers=4).to(device)
    
    try:
        v2_checkpoint = torch.load("./log/v2/best_model_v2.pt", map_location=device)
        cnn_v2.load_state_dict(v2_checkpoint['cnn_state_dict'])
        model_v2.load_state_dict(v2_checkpoint['model_state_dict'])
        print("Version 2 Transformer 로드 완료")
    except Exception as e:
        print(f"Version 2 모델 로드 실패: {e}")
        return
    
    # 사전 훈련된 모델들을 고정
    for param in cnn_v1.parameters():
        param.requires_grad = False
    for param in model_v1.parameters():
        param.requires_grad = False
    for param in cnn_v2.parameters():
        param.requires_grad = False
    for param in model_v2.parameters():
        param.requires_grad = False
    
    cnn_v1.eval()
    model_v1.eval()
    cnn_v2.eval()
    model_v2.eval()
    
    # 학습 가능한 앙상블 모델
    ensemble_model = TransformerEnsembleModel(
        cnn_v1, model_v1, cnn_v2, model_v2, 
        ensemble_method='learned'  # 학습 가능한 앙상블
    ).to(device)
    
    # Loss & Optimizer
    pos_weight = torch.tensor([1.2]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # 학습 가능한 파라미터만 옵티마이저에 추가
    trainable_params = []
    trainable_params.append(ensemble_model.ensemble_weights)
    trainable_params.extend(list(ensemble_model.ensemble_fc.parameters()))
    
    optimizer = torch.optim.AdamW(trainable_params, lr=5e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4, eta_min=1e-6)
    scaler = GradScaler()
    
    # 저장 경로
    best_model_path = f"{ensemble_dir}/best_speed_ensemble.pt"
    checkpoint_path = f"{ensemble_dir}/last_speed_checkpoint.pt"
    log_history = []

    print(f"\n📈 **속도 최적화 앙상블 설정**")
    print(f"   - 앙상블 방법: Learned Weighting")
    print(f"   - 배치 크기: {batch_size} × Accumulation {accumulation_steps} = Effective {batch_size * accumulation_steps}")
    print(f"   - 에포크: 4 (속도 우선)")
    print(f"   - Mixed Precision: 활성화")
    print(f"   - 예상 시간: 1-1.5시간/에포크")
    print("="*70)

    # 훈련 루프
    start_epoch = 0
    patience = 2
    patience_counter = 0
    best_val_loss = float('inf')
    num_epochs = 4  # 속도를 위해 에포크 수 줄임

    for epoch in range(start_epoch, num_epochs):
        current_lr = scheduler.get_last_lr()[0] if scheduler.get_last_lr() else 5e-4
        
        # 현재 앙상블 가중치 출력
        weights = torch.softmax(ensemble_model.ensemble_weights, dim=0).detach().cpu().numpy()
        
        print(f"\n[Epoch {epoch+1}/{num_epochs}] LR: {current_lr:.2e}, Weights: V1={weights[0]:.3f}, V2={weights[1]:.3f}")
        
        # 훈련 시간 측정
        train_start = time.time()
        train_loss = train_ensemble_speed(ensemble_model, train_loader, criterion, optimizer, device, scaler, accumulation_steps)
        train_time = time.time() - train_start
        
        # 검증 시간 측정 (전체 검증 데이터 사용)
        val_start = time.time()
        val_loss = validate_ensemble_speed(ensemble_model, val_loader, criterion, device)
        val_time = time.time() - val_start
        
        print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f} ({train_time/60:.1f}분)")
        print(f"[Epoch {epoch+1}] Val Loss: {val_loss:.4f} ({val_time/60:.1f}분)")

        # 빠른 성능 평가 (200 배치만 샘플링)
        eval_start = time.time()
        accuracy, recall, f1 = evaluate_ensemble_speed(ensemble_model, val_loader, device, threshold=0.7, max_batches=200)
        eval_time = time.time() - eval_start
        
        print(f"[Epoch {epoch+1}] Metrics: Acc={accuracy:.4f}, Rec={recall:.4f}, F1={f1:.4f} (⏱️{eval_time:.1f}초)")

        total_epoch_time = train_time + val_time + eval_time
        print(f"[Epoch {epoch+1}] 총 소요 시간: {total_epoch_time/60:.1f}분")

        log_history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "accuracy": accuracy,
            "recall": recall,
            "f1_score": f1,
            "learning_rate": current_lr,
            "v1_weight": weights[0],
            "v2_weight": weights[1],
            "train_time_min": train_time/60,
            "val_time_min": val_time/60,
            "total_time_min": total_epoch_time/60
        })

        # 베스트 모델 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'ensemble_state_dict': ensemble_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'accuracy': accuracy,
                'recall': recall,
                'f1_score': f1,
                'ensemble_method': 'learned',
                'ensemble_weights': ensemble_model.ensemble_weights.detach().cpu(),
                'training_time': total_epoch_time/60,
                'model_info': {
                    'v1_type': 'basic_transformer',
                    'v2_type': 'improved_transformer',
                    'v1_accuracy': 0.725,
                    'v2_accuracy': 0.769,
                    'ensemble_accuracy': accuracy,
                    'speed_optimized': True
                }
            }, best_model_path)
            print(f"Best model saved (Val Loss: {val_loss:.4f}, Acc: {accuracy:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"Early stopping patience {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"==== Early stopping Triggered ====")
                break

        # 체크포인트 저장
        torch.save({
            'ensemble_state_dict': ensemble_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch': epoch,
            'best_val_loss': best_val_loss
        }, checkpoint_path)
        
        scheduler.step()

    # 로그 저장
    log_df = pd.DataFrame(log_history)
    log_df.to_csv(f"{ensemble_dir}/speed_ensemble_log.csv", index=False)
    print(f"\n Training log saved to {ensemble_dir}/speed_ensemble_log.csv")

    # 최종 결과
    checkpoint = torch.load(best_model_path, map_location=device)
    final_accuracy = checkpoint['accuracy']
    final_recall = checkpoint['recall']
    final_f1 = checkpoint['f1_score']
    final_weights = checkpoint['ensemble_weights']
    
    print("\n" + "="*70)
    print(" **속도 최적화 앙상블 모델 완료!**")
    print("="*70)
    print(f"Version 1 (기본): 72.5% 정확도")
    print(f"Version 2 (개선): 76.9% 정확도")
    print(f"앙상블 (전체 데이터): {final_accuracy:.1%} 정확도")
    print(f"성능 향상: +{(final_accuracy - 0.769) * 100:.1f}%p (vs Version 2)")
    print(f"재현율: {final_recall:.1%}")
    print(f"F1-Score: {final_f1:.1%}")
    
    if final_weights is not None:
        weights = torch.softmax(final_weights, dim=0).numpy()
        print(f"학습된 최종 가중치: V1={weights[0]:.3f}, V2={weights[1]:.3f}")
    
    avg_time_per_epoch = log_df['total_time_min'].mean()
    print(f"평균 에포크 시간: {avg_time_per_epoch:.1f}분")
    print(f"모델 저장: {best_model_path}")
    
    if final_accuracy > 0.785:
        print("속도 최적화 + 성능 향상 성공!")
    elif final_accuracy > 0.77:
        print("속도와 성능의 균형잡힌 개선!")
    else:
        print("속도는 개선되었으나 성능 향상은 제한적")

if __name__ == '__main__':
    main()
