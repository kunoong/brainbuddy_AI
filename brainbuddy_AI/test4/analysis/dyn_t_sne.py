import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from collections import Counter

# === [1] 경로 설정 ===
dyn_dir = r"C:\eye_dataset\train2\dynamic_feature"  # ← 정확한 경로로 수정 필요

# === [2] 라벨 매핑 ===
def map_label(code):
    if code == "F":
        return "Focused"
    elif code == "N":
        return "Neglect"
    elif code in ["S", "D", "A"]:
        return "Other"
    else:
        return None  # Unknown 클래스 무시

color_map = {
    "Focused": "blue",
    "Neglect": "purple",
    "Other": "gray"
}

# === [3] 데이터 로딩 ===
all_data = []
all_labels = []

for file in os.listdir(dyn_dir):
    if not file.endswith(".csv"):
        continue

    try:
        label_code = file.split("_")[7]  # 8번째 위치에서 라벨 추출
        label = map_label(label_code)
        if label is None:
            continue

        path = os.path.join(dyn_dir, file)
        df = pd.read_csv(path)

        # CSV 한 줄 (1 sample) → 벡터로 flatten
        all_data.append(df.values.flatten())
        all_labels.append(label)
    except Exception as e:
        print(f"⚠️ 오류 - {file}: {e}")

# === [4] 클래스 분포 출력 ===
print("📊 클래스 분포:", Counter(all_labels))

# === [5] t-SNE 분석 ===
X = np.array(all_data)
y = np.array(all_labels)

# 표준화 후 t-SNE
X_scaled = StandardScaler().fit_transform(X)
X_tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42).fit_transform(X_scaled)

# === [6] 시각화 ===
plt.figure(figsize=(10, 8))
for label in sorted(set(y)):
    idx = y == label
    plt.scatter(X_tsne[idx, 0], X_tsne[idx, 1],
                label=label,
                c=color_map[label],
                alpha=0.7,
                s=40,
                edgecolors='k')

plt.title("t-SNE: Dynamic Feature Similarity (Focused / Neglect / Other)")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
