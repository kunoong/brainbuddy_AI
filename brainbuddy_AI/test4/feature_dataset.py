# import os
# import torch
# from torch.utils.data import Dataset
# import numpy as np
# import pandas as pd

# class FeatureDataset(Dataset):
#     LABEL_MAP = {'F': 0, 'S': 1, 'D': 2, 'A': 3, 'N': 4}

#     def __init__(self, seq_dir, dyn_dir, transform=None):
#         self.seq_dir = seq_dir
#         self.dyn_dir = dyn_dir
#         self.transform = transform

#         self.samples = []
#         for file in os.listdir(dyn_dir):
#             if not file.endswith("_dynamic.csv"):
#                 continue
#             prefix = file.replace("_dynamic.csv", "")
#             parts = prefix.split("_")
#             if len(parts) >= 8:
#                 label_char = parts[7]  # index 7 = 8번째 요소
#                 label = self.LABEL_MAP.get(label_char)
#                 if label is not None:
#                     seq_path = os.path.join(seq_dir, f"{prefix}.npy")
#                     dyn_path = os.path.join(dyn_dir, file)
#                     if os.path.exists(seq_path):
#                         self.samples.append((prefix, label))
#             else:
#                 print(f"⚠️ 라벨 추출 실패: {file}")

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         prefix, label = self.samples[idx]
#         seq_path = os.path.join(self.seq_dir, f"{prefix}.npy")
#         dyn_path = os.path.join(self.dyn_dir, f"{prefix}_dynamic.csv")

#         seq = np.load(seq_path).astype(np.float32)
#         dyn = pd.read_csv(dyn_path).values[0].astype(np.float32)

#         if self.transform:
#             seq = self.transform(seq)

#         return {
#             "sequence": torch.tensor(seq),       # [T, D]
#             "dynamic": torch.tensor(dyn),        
#             "label": torch.tensor(label).long()
#         }


# 이진 분류용
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class FeatureDataset(Dataset):
    # 집중(F): 1, 비집중(S, D, A, N): 0
    BINARY_LABEL_MAP = {'F': 1, 'S': 0, 'D': 0, 'A': 0, 'N': 0}

    def __init__(self, seq_dir, dyn_dir, transform=None):
        self.seq_dir = seq_dir
        self.dyn_dir = dyn_dir
        self.transform = transform

        self.samples = []
        for file in os.listdir(dyn_dir):
            if not file.endswith("_dynamic.csv"):
                continue
            prefix = file.replace("_dynamic.csv", "")
            parts = prefix.split("_")
            if len(parts) >= 8:
                label_char = parts[7]  # index 7 = 8번째 요소
                label = self.BINARY_LABEL_MAP.get(label_char)
                if label is not None:
                    seq_path = os.path.join(seq_dir, f"{prefix}.npy")
                    dyn_path = os.path.join(dyn_dir, file)
                    if os.path.exists(seq_path):
                        self.samples.append((prefix, label))
            else:
                print(f"⚠️ 라벨 추출 실패: {file}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        prefix, label = self.samples[idx]
        seq_path = os.path.join(self.seq_dir, f"{prefix}.npy")
        dyn_path = os.path.join(self.dyn_dir, f"{prefix}_dynamic.csv")

        seq = np.load(seq_path).astype(np.float32)
        dyn = pd.read_csv(dyn_path).values[0].astype(np.float32)

        if self.transform:
            seq = self.transform(seq)

        return {
            "sequence": torch.tensor(seq),       # [T, D]
            "dynamic": torch.tensor(dyn),        
            "label": torch.tensor(label).long()
        }
