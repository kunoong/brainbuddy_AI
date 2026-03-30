# 특징 중요도 파악하기
import torch
import numpy as np
from sklearn.metrics import accuracy_score
from copy import deepcopy
from tqdm import tqdm
from lstm import baseLSTMModel
from feature_dataset import FeatureDataset
from torch.utils.data import DataLoader

# === validation 데이터셋 로드
val_dataset = FeatureDataset(
    seq_dir="C:/eye_dataset/valid/lstm_seq",
    dyn_dir="C:/eye_dataset/valid/dynamic_feature"
)

val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

def compute_permutation_importance(model, val_loader, device, feature_names):
    model.eval()
    base_preds, base_labels = [], []

    # === 원본 정확도 구하기 ===
    with torch.no_grad():
        for batch in val_loader:
            x_seq = batch["sequence"].to(device)
            x_dyn = batch["dynamic"].to(device)
            labels = batch["label"].to(device)

            outputs = model(x_seq, x_dyn)
            preds = outputs.argmax(dim=1)
            base_preds.extend(preds.cpu().numpy())
            base_labels.extend(labels.cpu().numpy())

    base_acc = accuracy_score(base_labels, base_preds)
    print(f"Baseline Accuracy: {base_acc:.4f}")

    # === 각 feature importance 계산 ===
    importances = []
    for idx, feature_name in enumerate(feature_names):
        print(f"⏳ Permuting feature: {feature_name}")
        perm_preds, perm_labels = [], []

        for batch in val_loader:
            x_seq = batch["sequence"].to(device)
            x_dyn = batch["dynamic"].clone()
            labels = batch["label"].to(device)

            # 해당 feature만 셔플
            idx_tensor = torch.randperm(x_dyn.size(0))
            x_dyn[:, idx] = x_dyn[idx_tensor, idx]

            x_dyn = x_dyn.to(device)

            with torch.no_grad():
                outputs = model(x_seq, x_dyn)
                preds = outputs.argmax(dim=1)
                perm_preds.extend(preds.cpu().numpy())
                perm_labels.extend(labels.cpu().numpy())

        perm_acc = accuracy_score(perm_labels, perm_preds)
        drop = base_acc - perm_acc
        importances.append((feature_name, drop))
        print(f"  ➤ Acc drop: {drop:.4f}")

    # 정렬 후 반환
    importances.sort(key=lambda x: x[1], reverse=True)
    return importances


# 모델 로딩
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = baseLSTMModel(input_size=28, hidden_size=128, dynamic_size=7, num_classes=5)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device)

# feature 이름 순서 반드시 올바르게 지정
feature_names = [
    "blink_count", "blink_duration", "cam_distance_diff_smooth",
    "gaze_variance", "saccade_frequency", "fixation_duration", "head_stability"
]

# 중요도 계산
importances = compute_permutation_importance(model, val_loader, device, feature_names)

# 시각화 (선택)
import matplotlib.pyplot as plt

labels, drops = zip(*importances)
plt.figure(figsize=(8, 5))
plt.barh(labels, drops)
plt.xlabel("Accuracy Drop")
plt.title("Permutation Feature Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
