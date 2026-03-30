import numpy as np
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
import os
import pandas as pd

from matplotlib import font_manager, rc # 폰트 세팅을 위한 모듈 추가
font_path = "C:/Windows/Fonts/malgun.ttf" # 사용할 폰트명 경로 삽입
font = font_manager.FontProperties(fname = font_path).get_name()
rc('font', family = font)


# === 파일 불러오기
data = np.load("../log/shap_inputs_train.npz")
x_seq = data["x_seq"]
x_dyn = data["x_dyn"]
y = data["labels"]

X_all = np.concatenate([x_seq, x_dyn], axis=1)
# 1. 시퀀스 feature (38개)
seq_feature_names = [
    "head_pitch", "head_yaw", "head_roll",
    "l_eye_x", "l_eye_y", "r_eye_x", "r_eye_y",
    "l_EAR", "r_EAR",
    "gaze_x", "gaze_y", "gaze_z",
    "is_in_roi"
]

nan_mask = [f"{f}_nan" for f in seq_feature_names]
delta_features = [
    "delta_ear",
    "gaze_delta",
    "head_motion_delta",
    "eye_center_delta",
    "head_roll_velocity"
]
delta_mask = [f"{f}_nan" for f in delta_features]

# ✅ 38개
sequence_feature_names = seq_feature_names + nan_mask + delta_features + delta_mask

# 2. 동적 요약 feature (7개)
x_dyn_feature_names = [
    "blink_count",
    "blink_duration",
    "link_rate_change",
    "gaze_variance",
    "saccade_amplitude",
    "gaze_entropy",
    "fixation_dispersion",	
    "roi_dwell_time"
]

# ✅ 최종 45개
shap_feature_names = sequence_feature_names + x_dyn_feature_names

# 체크
assert len(shap_feature_names) == X_all.shape[1], "❌ feature 이름 수와 입력 차원 수가 불일치합니다!"

# === LightGBM 학습
model = lgb.LGBMClassifier()
model.fit(X_all, y)

# === SHAP 계산
explainer = shap.Explainer(model)
shap_values = explainer(X_all)  # shap_values.values shape: [N, num_classes, D]

print("X_all shape:", X_all.shape)  # (N, D)
print("shap_values.values shape:", shap_values.values.shape)  # (N, C, D?)
print("len(shap_feature_names):", len(shap_feature_names))

# === 클래스별 SHAP summary plot
os.makedirs("log/shap", exist_ok=True)
class_names = [0,1]

# === SHAP summary plot for binary classification
print(f"📊 SHAP 분석: 이진 분류 (Focused vs Unfocused)")

shap.summary_plot(
    shap_values.values,  # shape: (N, D)
    features=X_all,
    feature_names=shap_feature_names,
    show=False,
    plot_type="bar"
)

plt.title("SHAP Feature Importance - Binary Classification")
plt.tight_layout()
plt.savefig("../log/shap/shap_binary_summary.png")
plt.close()


# SHAP 값: 이진 분류인 경우 (N, D) 형태
shap_vals = np.abs(shap_values.values)  # (14015, 45)

# feature별 평균, 분산
mean_across_class = np.mean(shap_vals, axis=0)  # (45,)
var_across_class = np.var(shap_vals, axis=0)    # (45,)

# 디버그 출력
print("shap_vals shape:", shap_vals.shape)
print("mean_across_class shape:", mean_across_class.shape)
print("var_across_class shape:", var_across_class.shape)
print(len(mean_across_class))  # 이제는 45 출력돼야 함

# DataFrame 정리
df = pd.DataFrame({
    'feature': shap_feature_names,
    'mean_abs_shap': mean_across_class,
    'var_abs_shap': var_across_class
})

# 시각화: 평균 vs 분산
plt.figure(figsize=(10, 6))
plt.scatter(df['mean_abs_shap'], df['var_abs_shap'], alpha=0.8)

for i, row in df.iterrows():
    plt.text(row['mean_abs_shap'], row['var_abs_shap'], row['feature'], fontsize=8)

plt.xlabel("Mean |SHAP value| (평균 영향력)")
plt.ylabel("Variance of |SHAP value| (샘플 간 분산)")
plt.title("SHAP Feature 영향력 (평균 vs 분산)")
plt.grid(True)
plt.tight_layout()
plt.savefig("../log/shap/shap_summary_mean_vs_var.png")
plt.show()