import os
import numpy as np
import torch
import random
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 설정
SEQ_PATH = r"C:/eye_dataset/valid/lstm_seq"
LABEL_MAP = {'F': 0, 'S': 1, 'D': 2, 'A': 3, 'N': 4}
NUM_CLASSES = 5
REPEAT = 20  # 낮게 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 시드 고정
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Dataset
class StaticSequenceDataset(Dataset):
    def __init__(self, path):
        self.samples = []
        for fname in os.listdir(path):
            if not fname.endswith(".npy"): continue
            parts = fname.replace(".npy", "").split("_")
            if len(parts) < 8: continue
            label = LABEL_MAP.get(parts[7])
            if label is None: continue
            arr = np.load(os.path.join(path, fname))  # [30, 28]
            self.samples.append((arr, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        arr, label = self.samples[idx]
        return torch.tensor(arr, dtype=torch.float32), torch.tensor(label)

dataset = StaticSequenceDataset(SEQ_PATH)
loader = DataLoader(dataset, batch_size=32, shuffle=False)

# LSTM 모델
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn.squeeze(0))

model = LSTMModel(input_size=28, hidden_size=32, num_classes=NUM_CLASSES).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# 학습
for epoch in range(3):  # epoch도 낮게 설정
    model.train()
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()

# Permutation Importance
def permutation_importance_seq(model, loader, repeat=5):
    model.eval()
    with torch.no_grad():
        base_preds, base_labels = [], []
        for xb, yb in loader:
            preds = model(xb.to(device)).argmax(dim=1)
            base_preds.extend(preds.cpu().numpy())
            base_labels.extend(yb.numpy())
        base_acc = accuracy_score(base_labels, base_preds)
        print(f"Baseline Accuracy: {base_acc:.4f}")

    input_dim = next(iter(loader))[0].shape[-1]
    importances = []

    for i in range(input_dim):
        drops = []
        for _ in range(repeat):
            perm_preds, perm_labels = [], []
            for xb, yb in loader:
                xb_perm = xb.clone()
                idx = torch.randperm(xb.size(0))
                xb_perm[:, :, i] = xb_perm[idx, :, i]  # feature i만 섞기
                preds = model(xb_perm.to(device)).argmax(dim=1)
                perm_preds.extend(preds.cpu().numpy())
                perm_labels.extend(yb.numpy())
            perm_acc = accuracy_score(perm_labels, perm_preds)
            drops.append(base_acc - perm_acc)
        avg_drop = np.mean(drops)
        importances.append((f"feat_{i}", avg_drop))
        print(f"🔁 Feature {i}: Drop={avg_drop:.4f}")

    return sorted(importances, key=lambda x: x[1], reverse=True)

# 실행 및 시각화
importances = permutation_importance_seq(model, loader, repeat=REPEAT)
labels, drops = zip(*importances)

plt.figure(figsize=(10, 6))
plt.barh(labels, drops)
plt.xlabel("Accuracy Drop")
plt.title("Permutation Importance of Static Feature Sequences (LSTM Input)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
