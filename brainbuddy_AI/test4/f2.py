import os
import shutil
import json
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm

# === 설정 ===F
label_root = r"C:/Users/user/Downloads/126.디스플레이 중심 안구 움직임 영상 데이터/01-1.정식개방데이터/Training/02.라벨링데이터/TL"
output_root = r"C:/eye_dataset/train3"
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

#=== LSTM에 들어가는 동적 feature
def compute_dynamic_features_per_frame_delta(df):
    delta_feats = []
    prev = None
    for i in range(len(df)):
        current = df.iloc[i]
        if prev is None:
            delta_feats.append({
                "delta_ear": 0,
                "gaze_delta": 0,
                "head_motion_delta": 0,
                "eye_center_delta": 0,
                "head_roll_velocity": 0,
            })
        else:
            delta_ear = ((current["l_EAR"] + current["r_EAR"]) / 2.0) - \
                        ((prev["l_EAR"] + prev["r_EAR"]) / 2.0)

            gaze_delta = np.linalg.norm(current[["gaze_x", "gaze_y", "gaze_z"]].values -
                                        prev[["gaze_x", "gaze_y", "gaze_z"]].values)

            head_motion_delta = np.linalg.norm(current[["head_pitch", "head_yaw", "head_roll"]].values -
                                               prev[["head_pitch", "head_yaw", "head_roll"]].values)

            eye_center_now = np.mean([
                [current["l_eye_x"], current["l_eye_y"]],
                [current["r_eye_x"], current["r_eye_y"]]
            ], axis=0)
            eye_center_prev = np.mean([
                [prev["l_eye_x"], prev["l_eye_y"]],
                [prev["r_eye_x"], prev["r_eye_y"]]
            ], axis=0)
            eye_center_delta = np.linalg.norm(eye_center_now - eye_center_prev)

            # 추가된 head_roll_velocity
            head_roll_velocity = current["head_roll"] - prev["head_roll"]

            delta_feats.append({
                "delta_ear": delta_ear,
                "gaze_delta": gaze_delta,
                "head_motion_delta": head_motion_delta,
                "eye_center_delta": eye_center_delta,
                "head_roll_velocity": head_roll_velocity,
            })

        prev = current

    return pd.DataFrame(delta_feats)


def compute_dynamic_features(df):
    try:
        features = {}

        # === EAR 기반 blink
        l_ear = df["l_EAR"].fillna(0)
        r_ear = df["r_EAR"].fillna(0)
        ear = (l_ear + r_ear) / 2.0
        blink_threshold = 0.2
        is_blinking = ear < blink_threshold
        blink_groups = (is_blinking != is_blinking.shift(1)).cumsum()
        blink_count = is_blinking.groupby(blink_groups).sum().gt(0).sum()
        blink_duration = is_blinking.sum()
        blink_rate_change = np.abs(np.diff(is_blinking.astype(int))).sum() / len(df)

        # === Gaze Variance (3D)
        gaze_xyz = df[["gaze_x", "gaze_y", "gaze_z"]]
        gaze_variance = float(gaze_xyz.var().mean())

        # === Gaze Position (2D)
        gaze_x = df[["l_eye_x", "r_eye_x"]].mean(axis=1)
        gaze_y = df[["l_eye_y", "r_eye_y"]].mean(axis=1)
        gaze_xy = np.stack([gaze_x.values, gaze_y.values], axis=1)

        # === Saccade Amplitude
        gaze_diffs = np.linalg.norm(np.diff(gaze_xy, axis=0), axis=1)
        saccade_amplitude = float(np.mean(gaze_diffs))

        # === Gaze Entropy
        bins = np.histogram2d(gaze_x, gaze_y, bins=10)[0]
        prob = bins / np.sum(bins)
        prob = prob[prob > 0]
        gaze_entropy = -np.sum(prob * np.log(prob))

        # === Fixation Dispersion
        fixation_flags = gaze_diffs < 3
        fixation_points = gaze_xy[1:][fixation_flags]
        fixation_dispersion = np.std(fixation_points) if len(fixation_points) > 0 else 0.0

        # === ROI dwell time
        roi_dwell_time = df["is_in_roi"].sum() / len(df)

        # === Gaze 순차성 분석
        try:
            gaze_path_linearity = compute_gaze_linearity(gaze_xy)
            gaze_direction_reversals = compute_gaze_reversals(gaze_xy)
        except Exception as e:
            gaze_path_linearity = 0.0
            gaze_direction_reversals = 0

        # === 전체 feature 저장
        features.update({
            "blink_count": int(blink_count),
            "blink_duration": int(blink_duration),
            "blink_rate_change": float(blink_rate_change),
            "gaze_variance": gaze_variance,
            "saccade_amplitude": saccade_amplitude,
            "gaze_entropy": gaze_entropy,
            "fixation_dispersion": fixation_dispersion,
            "roi_dwell_time": roi_dwell_time,
            "gaze_path_linearity": gaze_path_linearity,
            "gaze_direction_reversals": gaze_direction_reversals
        })

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

#시선의 전체 경로 길이와, 처음~끝 직선 거리의 비율
def compute_gaze_linearity(gaze_xy):
    if len(gaze_xy) < 2:
        return 0.0

    total_path = np.sum(np.linalg.norm(np.diff(gaze_xy, axis=0), axis=1))
    straight_line = np.linalg.norm(gaze_xy[-1] - gaze_xy[0])
    return straight_line / total_path if total_path > 0 else 0.0

#시선 벡터의 방향이 얼마나 자주 뒤집히는지
def compute_gaze_reversals(gaze_xy):
    diffs = np.diff(gaze_xy, axis=0)
    if len(diffs) < 2:
        return 0
    unit_diffs = diffs / np.linalg.norm(diffs, axis=1, keepdims=True)
    dot_products = np.sum(unit_diffs[1:] * unit_diffs[:-1], axis=1)
    reversals = np.sum(dot_products < -0.5)  # 방향이 반대일 때
    return int(reversals)

def extract_features(json_path):
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        ann = data["Annotations"]
        label_dict = {item["label"]: item for item in ann["annotations"]}

        l_center = label_dict["l_center"]["points"][0]
        r_center = label_dict["r_center"]["points"][0]
        roll, pitch, yaw = ann["pose"]["head"]
        #cam_distance = ann["distance"]["cam"]

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
            #"cam_distance": cam_distance,
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
for seq in tqdm(range(41, 81), desc="시퀀스 처리"):
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
            # 정적 feature
            lstm_df = preprocess_for_lstm(df)

            # 변화율 기반 동적 feature
            dyn_seq_df = compute_dynamic_features_per_frame_delta(df)
            dyn_seq_df = preprocess_for_lstm(dyn_seq_df)

            # 정적 + 동적 feature concat
            combined_df = pd.concat([lstm_df, dyn_seq_df], axis=1)
            expected_dim = 36 
            if combined_df.shape[1] != expected_dim:
                print(f"❗ 결합 feature 수 불일치: {prefix} → {combined_df.shape}")
                continue

            # 저장
            np.save(os.path.join(output_seq_root, f"{prefix}.npy"), combined_df.to_numpy())

            # 평균 기반 동적 feature CSV 저장 (기존 유지)
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

#특징 이름 저장
with open(os.path.join(output_root, "feature_names.json"), "w", encoding="utf-8") as f:
    json.dump(list(combined_df.columns), f, indent=2)