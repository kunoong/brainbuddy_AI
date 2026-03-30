import os
import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

class VideoFolderDataset(Dataset):
    def __init__(self, data_list, transform=None, num_frames=30):
        self.num_frames = num_frames
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224), interpolation=InterpolationMode.BILINEAR, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.samples = []  # (sorted_paths:list[str], label)
        for folder_path, label in data_list:
            if not os.path.isdir(folder_path): continue
            files = [f for f in os.listdir(folder_path)
                     if f.lower().endswith(('.jpg','.jpeg','.png'))]
            if len(files) < self.num_frames: continue
            files.sort()
            paths = [os.path.join(folder_path, f) for f in files[:self.num_frames]]
            self.samples.append((paths, label))

        cv2.setNumThreads(0)# OpenCV 내부 스레드 비활성(멀티워커와 충돌/과다 스레딩 방지)

        # Normalize
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
        self.std  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)

    def __len__(self): return len(self.samples)

    def _load_frame(self, p):
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None: return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = Image.fromarray(img)
            img = self.transform(img)
            return img
        else:
            img = torch.from_numpy(img).permute(2,0,1).float()/255.0
            return img

    def __getitem__(self, idx):
        paths, label = self.samples[idx]
        frames = []
        last_ok = None
        for p in paths:
            t = self._load_frame(p)
            if t is None:
                t = last_ok if last_ok is not None else torch.zeros(3,224,224)
            else:
                last_ok = t
            frames.append(t)

        video = torch.stack(frames, dim=0)
        return video, torch.tensor(label, dtype=torch.float32)
