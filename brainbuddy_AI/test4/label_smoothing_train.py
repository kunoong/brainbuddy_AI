import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from tqdm import tqdm

from feature_dataset import FeatureDataset
from lstm import baseLSTMModel


class WeightedLabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.1, weight=None, reduction='mean'):
        super().__init__()
        self.eps = eps
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        num_classes = inputs.size(-1)
        log_preds = torch.log_softmax(inputs, dim=-1)
        if self.weight is not None:
            log_preds = log_preds * self.weight.unsqueeze(0)

        loss = -log_preds.sum(dim=-1)
        nll = -log_preds[range(inputs.size(0)), targets]
        loss = (1 - self.eps) * nll + self.eps * (loss / num_classes)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# class LabelSmoothingCrossEntropy(nn.Module):
#     def __init__(self, eps=0.1, reduction='mean'):
#         super(LabelSmoothingCrossEntropy, self).__init__()
#         self.eps = eps
#         self.reduction = reduction

#     def forward(self, inputs, targets):
#         num_classes = inputs.size(-1)
#         log_preds = torch.log_softmax(inputs, dim=-1)
#         loss = -log_preds.sum(dim=-1)
#         nll = -log_preds[range(inputs.size(0)), targets]
#         loss = (1 - self.eps) * nll + self.eps * (loss / num_classes)

#         if self.reduction == 'mean':
#             return loss.mean()
#         elif self.reduction == 'sum':
#             return loss.sum()
#         else:
#             return loss

# === 설정 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 5
input_size = 28
dynamic_size = 7
hidden_size = 128
batch_size = 32
num_epochs = 20
lr = 1e-3
val_ratio = 0.2
save_path = "best_model.pth"
log_path = "log/train_log_5.csv"
os.makedirs("log", exist_ok=True)


# === 데이터셋 불러오기
train_dataset = FeatureDataset(
    seq_dir="C:/eye_dataset/train/lstm_seq",
    dyn_dir="C:/eye_dataset/train/dynamic_feature"
)

val_dataset = FeatureDataset(
    seq_dir="C:/eye_dataset/valid/lstm_seq",
    dyn_dir="C:/eye_dataset/valid/dynamic_feature"
)
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import torch

# 모든 정답 레이블 추출
labels = [label for _, label in train_dataset.samples]

# 클래스 가중치 계산
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

print("📊 클래스별 가중치:", class_weights_tensor)


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# === 모델
model = baseLSTMModel(
    input_size=input_size,
    hidden_size=hidden_size,
    dynamic_size=dynamic_size,
    num_classes=num_classes
).to(device)

criterion = WeightedLabelSmoothingCrossEntropy(
    eps=0.05,  # 부드럽게 설정
    weight=class_weights_tensor
)

#criterion = LabelSmoothingCrossEntropy(eps=0.02)#criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# === 학습
best_val_acc = 0
log_rows = []
sample = next(iter(train_loader))
print("x_dyn shape:", sample["dynamic"].shape) 
for epoch in range(1, num_epochs + 1):
    model.train()
    train_loss, train_preds, train_labels = 0, [], []

    for batch in tqdm(train_loader, desc=f"[Epoch {epoch}] Training"):
        x_seq = batch["sequence"].to(device)
        x_dyn = batch["dynamic"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(x_seq, x_dyn)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_preds.extend(outputs.argmax(dim=1).cpu().numpy())
        train_labels.extend(labels.cpu().numpy())

    train_acc = accuracy_score(train_labels, train_preds)
    avg_train_loss = train_loss / len(train_loader)

    # === 검증
    model.eval()
    val_loss, val_preds, val_labels = 0, [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"[Epoch {epoch}] Validation"):
            x_seq = batch["sequence"].to(device)
            x_dyn = batch["dynamic"].to(device)
            labels = batch["label"].to(device)

            outputs = model(x_seq, x_dyn)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            val_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    val_acc = accuracy_score(val_labels, val_preds)
    avg_val_loss = val_loss / len(val_loader)

    print(f"📘 Epoch {epoch} | Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {avg_val_loss:.4f}, Acc: {val_acc:.4f}")

    # === 모델 저장
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), save_path)
        print(f"✅ 모델 저장됨 (best acc: {best_val_acc:.4f})")

    # === 로그 기록
    log_rows.append({
        "epoch": epoch,
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss,
        "train_acc": train_acc,
        "val_acc": val_acc
    })

# === CSV 로그 저장
df_log = pd.DataFrame(log_rows)
df_log.to_csv(log_path, index=False)
print(f"📄 학습 로그 저장 완료: {log_path}")

# === Confusion Matrix 저장
conf_mat = confusion_matrix(val_labels, val_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues",
            xticklabels=["F", "S", "D", "A", "N"],
            yticklabels=["F", "S", "D", "A", "N"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("log/conf_matrix_5.png")
plt.close()
print("🧩 confusion matrix 저장 완료 → log/conf_matrix_4.png")
