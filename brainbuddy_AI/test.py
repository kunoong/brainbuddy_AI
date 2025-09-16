# test_infer.py  (late fusion 없이 학습한 모델용 간단 테스트)
import os
import pickle
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from datasets.video_folder_dataset import VideoFolderDataset
from models.engagement_model import EngagementModel
from models.cnn_encoder import CNNEncoder

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    recall_score,
    f1_score
)

# ------------------ Utils ------------------
def load_data(pkl_files):
    all_data = []
    for pkl_path in pkl_files:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)  # [(folder_path, label), ...]
            all_data.extend(data)
    return all_data


# ------------------ Test only ------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_pkl_files = [
        # "C:/KSEB/brainbuddy_AI/preprocessing/pickle_labels/test/20_02.pkl",
        # "C:/KSEB/brainbuddy_AI/preprocessing/pickle_labels/test/20_04.pkl",
        "C:/KSEB/brainbuddy_AI/preprocessing/pickle_labels/test/our_dataset.pkl",
    ]
    best_model_path = "./log/train7/best_model/best_model_epoch_4.pt"  # 필요시 수정
    #best_model_path="log/train3_nolf/best_model/best_model_epoch_1.pt"
    
    # 데이터
    test_data_list = load_data(test_pkl_files)
    test_dataset = VideoFolderDataset(test_data_list)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=16, pin_memory=True)

    # 모델 로드
    cnn = CNNEncoder().to(device)
    model = EngagementModel().to(device)

    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Checkpoint not found: {best_model_path}")

    ckpt = torch.load(best_model_path, map_location=device)
    cnn.load_state_dict(ckpt['cnn_state_dict'])
    model.load_state_dict(ckpt['model_state_dict'])  # ✅ 이제 fc.0 크기 일치(64,128)

    cnn.eval()
    model.eval()

    # 추론 & 메트릭
    all_probs, all_preds, all_labels = [], [], []

    with torch.inference_mode():
        for videos, labels in tqdm(test_loader, desc="Test"):  # ✅ (videos, labels)만
            videos = videos.to(device)
            feats = cnn(videos)
            logits = model(feats)  # ✅ fusion 없음
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

    print(f"✅ Accuracy: {acc:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
    print("Confusion Matrix:\n", cm)

    # 확률 분포 시각화
    probs_class0 = [p for p, y in zip(all_probs, all_labels) if y == 0]
    probs_class1 = [p for p, y in zip(all_probs, all_labels) if y == 1]

    plt.figure()
    plt.hist(probs_class0, bins=20, alpha=0.6, label="Class 0",
            color="skyblue", edgecolor="black")
    plt.hist(probs_class1, bins=20, alpha=0.6, label="Class 1",
            color="salmon", edgecolor="black")
    plt.xlabel("Predicted probability (class=1)")
    plt.ylabel("Count")
    plt.title("Probability distribution by True Class")
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_dir = "./log/test"
    os.makedirs(save_dir, exist_ok=True)  # ← 베이스 폴더 보장
    prob_hist_path = os.path.join(save_dir, "prob_histogram_by_class.png")
    plt.savefig(prob_hist_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"📈 Probability histogram saved: {prob_hist_path}")
    
    # 혼동행렬 저장
    save_dir = "./log/test"
    os.makedirs(os.path.join(save_dir, "confusion_matrix"), exist_ok=True)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("[Test 1] Confusion Matrix")
    out_path = os.path.join(save_dir, "confusion_matrix", "conf_matrix_test.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"📊 Confusion matrix saved: {out_path}")


if __name__ == "__main__":
    main()
