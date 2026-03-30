import pickle
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def load_feature_file(path, label_value):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    feats = np.array(data['features'])  # [N, D]
    labels = np.array([label_value] * len(feats))
    return feats, labels

# --- Load AIHub 0, 1 레이블 분할된 파일 ---
aihub_1_feats, _ = load_feature_file('./cnn_features/features_30/train_20_01.pkl', label_value=1)
aihub_0_feats, _ = load_feature_file('./cnn_features/features_30/train_20_03.pkl', label_value=0)

# --- 평균 풀링 (시퀀스 → 벡터) ---
if aihub_1_feats.ndim == 3:
    aihub_1_feats = np.mean(aihub_1_feats, axis=1)
if aihub_0_feats.ndim == 3:
    aihub_0_feats = np.mean(aihub_0_feats, axis=1)

aihub_feats = np.concatenate([aihub_0_feats, aihub_1_feats], axis=0)
aihub_domain_label = np.array([0] * len(aihub_feats))  # AIHub: domain 0

# --- DAiSEE ---
with open('./cnn_features/features_30/D_train.pkl', 'rb') as f:
    daisee = pickle.load(f)
daisee_feats = np.array(daisee['features'])

if daisee_feats.ndim == 3:
    daisee_feats = np.mean(daisee_feats, axis=1)

daisee_domain_label = np.array([1] * len(daisee_feats))  # DAiSEE: domain 1


# --- Combine and t-SNE ---
X = np.concatenate([aihub_feats, daisee_feats], axis=0)
domain_labels = np.concatenate([aihub_domain_label, daisee_domain_label], axis=0)

X_embedded = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X)

# --- Plot ---
plt.figure(figsize=(8,6))
plt.scatter(X_embedded[domain_labels==0, 0], X_embedded[domain_labels==0, 1], c='blue', label='AIHub', alpha=0.5)
plt.scatter(X_embedded[domain_labels==1, 0], X_embedded[domain_labels==1, 1], c='red', label='DAiSEE', alpha=0.5)
plt.legend()
plt.title("t-SNE of CNN Features: AIHub (blue) vs DAiSEE (red)")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
plt.grid(True)
plt.show()
