import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from collections import Counter

# === 경로 설정 ===
seq_dir = "C:/eye_dataset/train2/lstm_seq"  # ← 수정 필요

# === 라벨 변환 매핑 (F / N / Other)
def convert_label(code):
    if code == "F":
        return "Focused"
    elif code == "N":
        return "Neglect"
    elif code in ["S", "D", "A"]:
        return "Other"
    else:
        return None  # 예외 처리

color_map = {
    "Focused": "blue",
    "Neglect": "purple",
    "Other": "gray"
}

# === 데이터 로딩 ===
all_data = []
all_labels = []

for file in os.listdir(seq_dir):
    if not file.endswith(".npy"):
        continue

    try:
        label_code = file.split("_")[7]  # 8번째 인덱스에서 클래스 추출
        label = convert_label(label_code)
        if label is None:
            continue

        data = np.load(os.path.join(seq_dir, file))
        pooled = data.mean(axis=0)
        all_data.append(pooled)
        all_labels.append(label)
    except Exception as e:
        print(f"⚠️ 오류 - {file}: {e}")

# === 클래스 분포 확인 ===
print("📊 3분류 클래스 분포:", Counter(all_labels))

# === t-SNE 계산 ===
X = np.array(all_data)
y = np.array(all_labels)
X_scaled = StandardScaler().fit_transform(X)
X_tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42).fit_transform(X_scaled)

# === 시각화 ===
plt.figure(figsize=(10, 8))

for label in sorted(set(y)):
    idx = y == label
    plt.scatter(X_tsne[idx, 0], X_tsne[idx, 1],
                label=label,
                c=color_map[label],
                alpha=0.4,
                s=40,
                edgecolors='k')

plt.title("t-SNE: 3-Class Separation (Focused / Neglect / Other)")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
