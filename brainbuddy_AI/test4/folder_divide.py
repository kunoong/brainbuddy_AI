import os
import shutil
import json
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm

# === 설정 ===F
label_root = r"C:/Users/user/Downloads/126.디스플레이 중심 안구 움직임 영상 데이터/01-1.정식개방데이터/Training/02.라벨링데이터/TL"
output_root = r"C:/eye_dataset/train"
output_seq_root = os.path.join(output_root, "lstm_seq")
output_dyn_root = os.path.join(output_root, "dynamic_feature")
os.makedirs(output_seq_root, exist_ok=True)
os.makedirs(output_dyn_root, exist_ok=True)

devices = ["Monitor", "Laptop"]
json_subdir = "json_rgb"
max_count = 30
min_count = 11

check_labels = ["l_center", "r_center", "l_eyelid", "r_eyelid", "l_iris", "r_iris"]
check_pose_fields = ["head", "cam", "point"]
check_distance_fields = ["cam"]

missing_label_files = []
log_entries = []

# === 전처리: NaN 보간 + 마스크
def preprocess_for_lstm(df, seq_len=30):
    nan_mask = df.isna().astype(int)
    nan_mask.columns = [f"{col}_nan" for col in df.columns]

    df.interpolate(method='linear', limit_direction='both', inplace=True)
    df.fillna(0, inplace=True)

    df = pd.concat([df, nan_mask], axis=1)

    if len(df) < seq_len:
        pad_df = pd.DataFrame([df.iloc[-1].to_dict()] * (seq_len - len(df)))
        df = pd.concat([df, pad_df], ignore_index=True)
    else:
        df = df.iloc[:seq_len].reset_index(drop=True)

    return df

# === 동적 feature 계산
def compute_dynamic_features(df):
    try:
        features = {}

        # === 1. Blink Count & Duration ===
        # EAR(Eye Aspect Ratio) 기준: blink 발생은 EAR < 0.2 (Soukupová & Čech, 2016)
        l_ear = df["l_EAR"].fillna(0)
        r_ear = df["r_EAR"].fillna(0)
        ear = (l_ear + r_ear) / 2.0
        blink_threshold = 0.2
        is_blinking = ear < blink_threshold
        blink_groups = (is_blinking != is_blinking.shift(1)).cumsum()
        blink_count = is_blinking.groupby(blink_groups).sum().gt(0).sum()
        blink_duration = is_blinking.sum()  # total frames blinking

        features["blink_count"] = int(blink_count)
        features["blink_duration"] = int(blink_duration)

        # === 2. Cam Distance Smooth Diff ===
        cam_distance = df["cam_distance"].ffill().bfill()
        cam_diff = cam_distance.diff().abs().rolling(window=5).mean()
        features["cam_distance_diff_smooth"] = float(cam_diff.mean())

        # === 3. Gaze Variance ===
        gaze_xyz = df[["gaze_x", "gaze_y", "gaze_z"]]
        features["gaze_variance"] = float(gaze_xyz.var().mean())  # variance over time

        # === 4. Saccade Frequency ===
        # 기준: 시선 변화량이 100 px/sec 이상이면 saccade (Holmqvist et al., 2011)
        gaze_xy = df[["l_eye_x", "l_eye_y", "r_eye_x", "r_eye_y"]].mean(axis=1)
        diffs = np.sqrt(np.diff(gaze_xy.values) ** 2)
        saccade_frequency = np.sum(diffs > 10)  # assuming ~30Hz, 10px change = saccade
        features["saccade_frequency"] = int(saccade_frequency)

        # === 5. Fixation Duration ===
        # 기준: 200ms 이상 동일 위치 응시는 fixation (Duchowski, 2007)
        fixation_flags = diffs < 3  # threshold for fixation
        groups = (fixation_flags != np.roll(fixation_flags, 1)).cumsum()
        fixation_durations = pd.Series(fixation_flags).groupby(groups).sum()
        longest_fixation = fixation_durations.max() if not fixation_durations.empty else 0
        features["fixation_duration"] = int(longest_fixation)

        # === 6. Head Stability ===
        # 기준: 머리 회전의 표준편차가 낮을수록 안정적 (표준 HCI 정의)
        head_motion_std = df[["head_pitch", "head_yaw", "head_roll"]].std().mean()
        features["head_stability"] = float(head_motion_std)

        return features

    except Exception as e:
        print(f"❌ 동적 feature 계산 실패: {e}")
        return {}


# === 필수 라벨 및 좌표 유효성 검사
def get_missing_fields(json_path):
    missing = []
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        ann = data.get("Annotations", {})
        annotations = ann.get("annotations", [])
        labels = [a["label"] for a in annotations]

        for label in check_labels:
            if label not in labels:
                missing.append(f"label:{label}")
            else:
                obj = next((a for a in annotations if a["label"] == label), {})
                if not obj.get("points") or not isinstance(obj["points"], list) or len(obj["points"]) == 0:
                    missing.append(f"{label}_empty_points")

        pose = ann.get("pose", {})
        for field in check_pose_fields:
            if field not in pose:
                missing.append(f"pose.{field}")

        dist = ann.get("distance", {})
        for field in check_distance_fields:
            if field not in dist:
                missing.append(f"distance.{field}")
    except Exception:
        missing.append("json_error")
    return missing

# === EAR 계산 ===
def compute_ear(points):
    if len(points) < 6:
        return np.nan
    p = np.array(points)
    vertical = np.linalg.norm(p[1] - p[5])
    horizontal = np.linalg.norm(p[0] - p[3])
    return vertical / horizontal if horizontal > 0 else np.nan

# === 시선 벡터 계산 ===
def compute_gaze_vector(l_eye, r_eye):
    vec = np.array([r_eye[0] - l_eye[0], r_eye[1] - l_eye[1], 1.0])
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else [0, 0, 1]

# === ROI 계산
def is_in_display_area(point, bounds=(0, 0, 1920, 1080)):
    x, y = point
    return int(bounds[0] <= x <= bounds[2] and bounds[1] <= y <= bounds[3])

def extract_features(json_path):
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        ann = data["Annotations"]
        label_dict = {item["label"]: item for item in ann["annotations"]}

        l_center = label_dict["l_center"]["points"][0]
        r_center = label_dict["r_center"]["points"][0]
        roll, pitch, yaw = ann["pose"]["head"]
        cam_distance = ann["distance"]["cam"]

        # === EAR 계산
        l_eyelid_points = label_dict.get("l_eyelid", {}).get("points", [])
        r_eyelid_points = label_dict.get("r_eyelid", {}).get("points", [])
        l_ear = compute_ear(l_eyelid_points)
        r_ear = compute_ear(r_eyelid_points)

        # === Gaze 벡터 계산
        gaze_x, gaze_y, gaze_z = compute_gaze_vector(l_center, r_center)

        # === ROI 계산
        gaze_point = ann["pose"].get("point", [960, 540])
        is_in_roi = is_in_display_area(gaze_point)

        return {
            "head_pitch": pitch,
            "head_yaw": yaw,
            "head_roll": roll,
            "cam_distance": cam_distance,
            "l_eye_x": l_center[0],
            "l_eye_y": l_center[1],
            "r_eye_x": r_center[0],
            "r_eye_y": r_center[1],
            "l_EAR": l_ear,
            "r_EAR": r_ear,
            "gaze_x": gaze_x,
            "gaze_y": gaze_y,
            "gaze_z": gaze_z,
            "is_in_roi": is_in_roi
        }
    except Exception as e:
        print(f"⚠️ feature 추출 실패: {e}")
        return None

# === 메인 루프
for seq in tqdm(range(1, 149), desc="시퀀스 처리"):
    seq_str = f"{seq:03d}"
    for device in devices:
        json_dir = os.path.join(label_root, seq_str, "T1", device, json_subdir)
        if not os.path.exists(json_dir):
            continue

        json_files = glob(os.path.join(json_dir, "*.json"))
        if not json_files:
            continue

        label_groups = {}
        for file in json_files:
            filename = os.path.basename(file)
            parts = filename.split("_")
            if len(parts) < 4:
                continue
            posture = parts[-3]
            if posture not in ["C", "D", "H", "T", "U"]:
                continue
            prefix = "_".join(parts[:-1])
            label_groups.setdefault(prefix, []).append(file)

        for prefix, files in label_groups.items():
            files.sort(key=lambda x: int(os.path.splitext(x)[0].split("_")[-1]))

            valid_files = [f for f in files if not get_missing_fields(f)]
            if len(valid_files) <= min_count:
                print(f"⚠️ 건너뜀 (유효 라벨 {len(valid_files)}개 이하): {prefix}")
                continue

            # 시퀀스 정규화
            if len(valid_files) < max_count:
                front = valid_files[:2] if len(valid_files) >= 2 else valid_files[:1]
                back = valid_files[-2:] if len(valid_files) >= 2 else valid_files[-1:]
                extended = front + valid_files + back
                while len(extended) < max_count:
                    extended.append(valid_files[-1])
            else:
                extended = valid_files[:max_count]

            # 저장 경로
            target_folder = os.path.join(output_root, prefix)
            os.makedirs(target_folder, exist_ok=True)

            features = []
            for i, f in enumerate(extended):
                target_path = os.path.join(target_folder, f"{i:03d}.json")
                shutil.copy2(f, target_path)
                feat = extract_features(f)
                if feat:
                    features.append(feat)

            if len(features) < 5:
                print(f"⚠️ feature 추출 실패 (5개 미만): {prefix}")
                continue

            df = pd.DataFrame(features)
            lstm_df = preprocess_for_lstm(df)
            if lstm_df.shape[1] != 38:
                print(f"❗ feature 수 불일치: {prefix} → {lstm_df.shape}")
                continue
            np.save(os.path.join(output_seq_root, f"{prefix}.npy"), lstm_df.to_numpy())
            #print(f"✅ 시퀀스 저장: {prefix} ({len(lstm_df)} rows)")

            # 💡 동적 feature 저장
            dyn_feats = compute_dynamic_features(df)
            if dyn_feats:
                dyn_df = pd.DataFrame([dyn_feats])
                dyn_df.to_csv(os.path.join(output_dyn_root, f"{prefix}_dynamic.csv"), index=False)

# 로그 저장
if missing_label_files:
    log_path = os.path.join(output_root, "missing_fields_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"총 누락 JSON 수: {len(missing_label_files)}\n")
        for path in missing_label_files:
            f.write(path + "\n")
    print(f"\n📄 누락된 항목 기록 저장 완료 → {log_path}")
else:
    print("\n✅ 모든 JSON 파일에 필수 필드 및 좌표가 존재합니다.")
