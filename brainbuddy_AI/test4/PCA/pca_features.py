import os
import numpy as np
import pandas as pd
from glob import glob
from sklearn.decomposition import PCA

# 설정
input_seq_dir = "C:/eye_dataset/train1/lstm_seq"
output_seq_dir = "C:/eye_dataset/train1/lstm_seq_pca"
os.makedirs(output_seq_dir, exist_ok=True)

# PCA 그룹 정의
eye_features = ['l_eye_x', 'l_eye_y', 'r_eye_x', 'r_eye_y', 'l_EAR', 'r_EAR', 'delta_ear']
head_features = ['head_pitch', 'head_yaw', 'head_roll', 'head_motion_delta']
gaze_features = ['gaze_x', 'gaze_y', 'gaze_z', 'gaze_delta']

# 단독 유지
retain_features = [
    'eye_center_delta',
    'gaze_variance',
    'saccade_frequency',
    'fixation_duration',
    'cam_distance', 'cam_dist_delta',
    'is_in_roi'
]
# 👇 원래 feature 이름 리스트
sequence_feature_names = [
    "head_pitch", "head_yaw", "head_roll",
    "cam_distance",
    "l_eye_x", "l_eye_y", "r_eye_x", "r_eye_y",
    "l_EAR", "r_EAR",
    "gaze_x", "gaze_y", "gaze_z",
    "is_in_roi"
]

nan_mask = [f"{f}_nan" for f in sequence_feature_names]
delta_features = [
    "delta_ear", "cam_dist_delta", "gaze_delta", "head_motion_delta", "eye_center_delta"
]
delta_mask = [f"{f}_nan" for f in delta_features]

# 전체 38개 이름
all_feature_names = sequence_feature_names + nan_mask + delta_features + delta_mask

# nan mask 자동 포함
def mask_names(cols): return [f"{c}_nan" for c in cols]
eye_mask = mask_names(eye_features)
head_mask = mask_names(head_features)
gaze_mask = mask_names(gaze_features)
delta_mask = mask_names(['delta_ear', 'cam_dist_delta', 'gaze_delta', 'head_motion_delta', 'eye_center_delta'])

# PCA 모델 준비
eye_pca = PCA(n_components=2)
head_pca = PCA(n_components=1)
gaze_pca = PCA(n_components=2)

# 기준 데이터 (첫 샘플)로 PCA 피팅
sample_path = glob(os.path.join(input_seq_dir, "*.npy"))[0]
sample_data = np.load(sample_path)  # [30, D]
df = pd.DataFrame(sample_data, columns=all_feature_names)


eye_pca.fit(df[eye_features])
head_pca.fit(df[head_features])
gaze_pca.fit(df[gaze_features])

# 전체 반복
for file_path in glob(os.path.join(input_seq_dir, "*.npy")):
    data = np.load(file_path)
    df = pd.DataFrame(sample_data, columns=all_feature_names)
    # PCA 변환
    eye_pca_vals = eye_pca.transform(df[eye_features])
    head_pca_vals = head_pca.transform(df[head_features])
    gaze_pca_vals = gaze_pca.transform(df[gaze_features])

    new_df = pd.DataFrame({
        "eye_pca_1": eye_pca_vals[:, 0],
        "eye_pca_2": eye_pca_vals[:, 1],
        "head_pca_1": head_pca_vals[:, 0],
        "gaze_pca_1": gaze_pca_vals[:, 0],
        "gaze_pca_2": gaze_pca_vals[:, 1],
    })

    # 나머지 보존 feature들 추가
    retained_cols = retain_features + eye_mask + head_mask + gaze_mask + delta_mask
    for col in retained_cols:
        if col in df.columns:
            new_df[col] = df[col]
        else:
            new_df[col] = 0  # 누락된 마스크 채움

    np.save(os.path.join(output_seq_dir, os.path.basename(file_path)), new_df.to_numpy())

print("✅ PCA 변환 및 저장 완료")
# === VALID SET 변환 ===
valid_input_seq_dir = "C:/eye_dataset/valid1/lstm_seq"
valid_output_seq_dir = "C:/eye_dataset/valid1/lstm_seq_pca"
os.makedirs(valid_output_seq_dir, exist_ok=True)

for file_path in glob(os.path.join(valid_input_seq_dir, "*.npy")):
    data = np.load(file_path)
    df = pd.DataFrame(sample_data, columns=all_feature_names)

    # PCA 변환 (train에서 학습한 모델 사용)
    eye_pca_vals = eye_pca.transform(df[eye_features])
    head_pca_vals = head_pca.transform(df[head_features])
    gaze_pca_vals = gaze_pca.transform(df[gaze_features])

    new_df = pd.DataFrame({
        "eye_pca_1": eye_pca_vals[:, 0],
        "eye_pca_2": eye_pca_vals[:, 1],
        "head_pca_1": head_pca_vals[:, 0],
        "gaze_pca_1": gaze_pca_vals[:, 0],
        "gaze_pca_2": gaze_pca_vals[:, 1],
    })

    # 나머지 보존 feature들 추가
    for col in retained_cols:
        if col in df.columns:
            new_df[col] = df[col]
        else:
            new_df[col] = 0

    np.save(os.path.join(valid_output_seq_dir, os.path.basename(file_path)), new_df.to_numpy())

print("✅ VALID PCA 변환 및 저장 완료")
