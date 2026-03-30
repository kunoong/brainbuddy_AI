# 조명, 색상 분포 차이가 클래스에 따라 다른지 확인
import os
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# 폴더명에서 라벨 추출: 끝 2자리
def extract_label_from_folder(folder_name):
    label_code = folder_name[-2:]
    return 1 if label_code == '01' else 0  # 01: 집중(1), 03: 비집중(0)

# 밝기 및 RGB 평균 추출
def compute_brightness_and_color(segment_path, resize=(128, 128)):
    files = sorted([f for f in os.listdir(segment_path) if f.endswith('.jpg')])
    brightness_list = []
    r_list, g_list, b_list = [], [], []

    for f in files:
        img = cv2.imread(os.path.join(segment_path, f))
        if img is None:
            continue
        img = cv2.resize(img, resize)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness_list.append(np.mean(gray))

        r_list.append(np.mean(img[:, :, 2]))  # R 채널
        g_list.append(np.mean(img[:, :, 1]))  # G 채널
        b_list.append(np.mean(img[:, :, 0]))  # B 채널

    return {
        "brightness": np.mean(brightness_list) if brightness_list else 0,
        "r": np.mean(r_list) if r_list else 0,
        "g": np.mean(g_list) if g_list else 0,
        "b": np.mean(b_list) if b_list else 0
    }

# 기본 데이터 경로
base_dir = r"C:/AIhub_frames/train"

# 데이터프레임용 리스트
records = []

# 시퀀스 폴더 리스트
folder_list = os.listdir(base_dir)

# 데이터 수집
for folder_name in tqdm(folder_list, desc="Extracting brightness and color stats"):
    folder_path = os.path.join(base_dir, folder_name)

    if not os.path.exists(folder_path):
        continue

    label = extract_label_from_folder(folder_name)
    for segment_name in os.listdir(folder_path):
        segment_path = os.path.join(folder_path,segment_name)
        if not os.path.isdir(segment_path):
            continue
    
        stats = compute_brightness_and_color(segment_path)

        records.append({
            "sequence": folder_name,
            "label": label,
            "brightness": stats["brightness"],
            "r": stats["r"],
            "g": stats["g"],
            "b": stats["b"]
        })

# DataFrame으로 변환
df = pd.DataFrame(records)

# 결과 저장 (선택)
df.to_csv("sequence_color_brightness_stats.csv", index=False)

# ================================================
# 📊 시각화
# ================================================

# 라벨 이름 매핑
label_map = {0: "Non-Focus", 1: "Focus"}
df['label_name'] = df['label'].map(label_map)

# Boxplot - 밝기
plt.figure(figsize=(6, 5))
sns.boxplot(x='label_name', y='brightness', data=df, palette={"Focus": "blue", "Non-Focus": "orange"})
plt.title("Brightness Distribution by Class")
plt.xlabel("Class")
plt.ylabel("Brightness")
plt.grid(True)
plt.tight_layout()
plt.show()

# Boxplot - RGB 채널
plt.figure(figsize=(12, 4))
for i, color in enumerate(['r', 'g', 'b']):
    plt.subplot(1, 3, i+1)
    sns.boxplot(x='label_name', y=color, data=df, palette={"Focus": "blue", "Non-Focus": "orange"})
    plt.title(f"{color.upper()} Channel by Class")
    plt.xlabel("Class")
    plt.ylabel(f"{color.upper()} Mean")
    plt.grid(True)
plt.tight_layout()
plt.show()
