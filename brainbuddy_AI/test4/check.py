import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob

# === 설정 ===
lstm_dir = r"C:/eye_dataset/train/lstm_seq"
dyn_dir = r"C:/eye_dataset/train/dynamic_feature"

# === 모든 npy 파일 불러오기
lstm_files = glob(os.path.join(lstm_dir, "*.npy"))

all_lstm_data = []
for path in lstm_files:
    data = np.load(path)
    all_lstm_data.append(data)

# === 병합
lstm_concat = np.concatenate(all_lstm_data, axis=0)

# === 임시 컬럼 이름 설정 (26개)
feature_names = [
    "head_pitch", "head_yaw", "head_roll", "cam_distance",
    "l_eye_x", "l_eye_y", "r_eye_x", "r_eye_y",
    "l_EAR", "r_EAR", "gaze_x", "gaze_y", "gaze_z"
]
# 나머지는 NaN 마스크 컬럼
feature_names += [f"{col}_nan" for col in feature_names]

df_lstm = pd.DataFrame(lstm_concat, columns=feature_names)

# === 분포 시각화
selected_cols = ["head_pitch", "head_yaw", "cam_distance", "l_EAR", "r_EAR", "gaze_x", "gaze_y", "gaze_z"]
plt.figure(figsize=(18, 12))
for i, col in enumerate(selected_cols):
    plt.subplot(3, 3, i+1)
    sns.histplot(df_lstm[col], kde=True, bins=50)
    plt.title(f"Distribution of {col}")
plt.tight_layout()
plt.show()

# === 요약 통계 출력
print("\n📊 Feature Summary Stats:")
print(df_lstm[selected_cols].describe())
