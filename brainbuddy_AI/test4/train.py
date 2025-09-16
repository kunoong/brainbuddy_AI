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
from lstm import BiLSTMAttnModel

# === 설정 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 2#5
input_size = 36 # PCA안했을 때 : 38, PCA 적용시 : 29
dynamic_size = 8
hidden_size = 128
batch_size = 32
num_epochs = 20
lr = 1e-3
val_ratio = 0.2
save_path = "best_model.pth"
log_path = "log/train_log_10(feat).csv"
os.makedirs("log", exist_ok=True)


# === 데이터셋 불러오기
train_dataset = FeatureDataset(
    #seq_dir="C:/eye_dataset/train1/lstm_seq",
    seq_dir="C:/eye_dataset/train2/lstm_seq",
    dyn_dir="C:/eye_dataset/train2/dynamic_feature"
)

val_dataset = FeatureDataset(
    #seq_dir="C:/eye_dataset/valid1/lstm_seq",
    seq_dir="C:/eye_dataset/valid2/lstm_seq",
    dyn_dir="C:/eye_dataset/valid2/dynamic_feature"
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# === 모델
model = BiLSTMAttnModel(
    input_size=input_size,
    hidden_size=hidden_size,
    dynamic_size=dynamic_size,
    num_classes=num_classes
).to(device)

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        """
        alpha: class imbalance 보정용 가중치 (ex. [1.0, 5.0])
        gamma: focusing parameter
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha if isinstance(alpha, torch.Tensor) else torch.tensor(alpha)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        CE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-CE_loss)

        if self.alpha is not None:
            at = self.alpha.gather(0, targets)
            CE_loss = at * CE_loss

        focal_loss = ((1 - pt) ** self.gamma) * CE_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

#criterion = nn.CrossEntropyLoss()
# class_weights = torch.tensor([1.0, 4.0], dtype=torch.float).to(device)
# criterion = nn.CrossEntropyLoss(weight=class_weights)
alpha = torch.tensor([1.0, 4.0]).to(device)  # [Unfocused, Focused]
criterion = FocalLoss(alpha=alpha, gamma=2.0)
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

def save_shap_inputs(dataloader, filename="shap_inputs_train.npz"):
    all_seq = []
    all_dyn = []
    all_labels = []

    for batch in dataloader:
        x_seq = batch["sequence"]  # [B, T, D]
        x_dyn = batch["dynamic"]   # [B, D_dyn]
        y = batch["label"]

        x_seq_mean = x_seq.mean(dim=1)  # [B, D]
        all_seq.append(x_seq_mean.numpy())
        all_dyn.append(x_dyn.numpy())
        all_labels.append(y.numpy())

    np.savez("log/" + filename,
             x_seq=np.concatenate(all_seq, axis=0),
             x_dyn=np.concatenate(all_dyn, axis=0),
             labels=np.concatenate(all_labels, axis=0))


# === CSV 로그 저장
df_log = pd.DataFrame(log_rows)
df_log.to_csv(log_path, index=False)
print(f"📄 학습 로그 저장 완료: {log_path}")

# === Confusion Matrix 저장
conf_mat = confusion_matrix(val_labels, val_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues",
            # xticklabels=["F", "S", "D", "A", "N"],
            # yticklabels=["F", "S", "D", "A", "N"]
            xticklabels=["Unfocused", "Focused"],
            yticklabels=["Unfocused", "Focused"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("log/conf_matrix_10(feat).png")
plt.close()
print("🧩 confusion matrix 저장 완료 → log/conf_matrix_10(feat).png")

#== shap
save_shap_inputs(train_loader, filename="shap_inputs_train.npz")
save_shap_inputs(val_loader, filename="shap_inputs_val.npz")

# === 예측 확률 분포 시각화 ===
def plot_prediction_distribution(dataloader, model, save_path="log/pred_dist.png"):
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            x_seq = batch["sequence"].to(device)
            x_dyn = batch["dynamic"].to(device)
            labels = batch["label"].to(device)

            outputs = model(x_seq, x_dyn)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # 클래스 1 (Focused)에 대한 확률만
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # 히스토그램
    plt.figure(figsize=(8, 5))
    plt.hist(all_probs[all_labels == 1], bins=50, alpha=0.7, label="Focused (label=1)")
    plt.hist(all_probs[all_labels == 0], bins=50, alpha=0.7, label="Unfocused (label=0)")
    plt.axvline(x=0.5, color='red', linestyle='--', label='Threshold = 0.5')
    plt.xlabel("Predicted Probability of Focused (Class 1)")
    plt.ylabel("Count")
    plt.title("Prediction Probability Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    # 확률 통계 출력
    print("📊 Focused 클래스 예측 확률 평균:", all_probs[all_labels == 1].mean())
    print("📊 Unfocused 클래스 예측 확률 평균:", all_probs[all_labels == 0].mean())

plot_prediction_distribution(val_loader, model, save_path="log/pred_dist.png")
