import torch
import torch.nn as nn
from torchvision import models
import os

# ==========================================
# 1. 모델 구조 정의 (기존과 동일)
# ==========================================
class CNNEncoder(nn.Module):
    def __init__(self, backbone="resnet18"):
        super().__init__()
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(*list(m.children())[:-1])
        self.out_dim = 512

    def forward(self, x):  
        f = self.encoder(x)
        return f.view(f.size(0), -1)

class CNN_LSTM(nn.Module):
    def __init__(self, backbone="resnet18", hidden=256, num_layers=2, bidirectional=True, dropout=0.3):
        super().__init__()
        self.cnn = CNNEncoder(backbone=backbone)
        self.lstm = nn.LSTM(
            input_size=self.cnn.out_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            dropout=0.0 if num_layers==1 else 0.2,
            bidirectional=bidirectional,
            batch_first=True,
        )
        d = 2 if bidirectional else 1
        self.head = nn.Sequential(
            nn.Linear(hidden*d, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        
    def forward(self, x):  
        B,T,C,H,W = x.shape
        x = x.reshape(B*T, C, H, W)
        feats = self.cnn(x).view(B, T, -1)
        seq, _ = self.lstm(feats)
        pooled = seq.mean(dim=1)
        return self.head(pooled).squeeze(1)

class ServingWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        logits = self.model(x)
        return torch.sigmoid(logits)

# ==========================================
# 2. ONNX 추출 로직 (경로 수정 완료)
# ==========================================
def export_to_onnx():
    print("🚀 ONNX 변환을 시작합니다...")
    
    base_model = CNN_LSTM(backbone="resnet18", hidden=256, num_layers=2, bidirectional=True)
    model = ServingWrapper(base_model)
    model.eval()

    dummy_input = torch.randn(1, 30, 3, 224, 224)

    # 🚨 경로 수정: 윈도우 환경에서 가장 안전한 절대 경로 방식 사용
    # 현재 실행 중인 파일(export_onnx.py)의 위치를 기준으로 brainbuddy_AI/onnx_model 폴더 지정
    current_dir = os.path.dirname(os.path.abspath(__file__)) # scripts 폴더
    project_root = os.path.dirname(current_dir)             # brainbuddy_AI 폴더
    save_dir = os.path.join(project_root, "onnx_model")
    
    # 폴더가 없으면 생성
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"📂 폴더 생성됨: {save_dir}")

    onnx_path = os.path.join(save_dir, "brainbuddy_random.onnx")

    # ONNX 내보내기 (opset_version을 에러 권장 사항인 18로 상향)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=18,  # 최신 환경에 맞춰 18로 변경
        do_constant_folding=True,
        input_names=['input_frames'],
        output_names=['engagement_prob'],
        dynamic_axes={
            'input_frames': {0: 'batch_size'},
            'engagement_prob': {0: 'batch_size'}
        }
    )
    
    print("-" * 50)
    print(f"✅ 성공! ONNX 모델이 아래 경로에 저장되었습니다:")
    print(f"📍 {onnx_path}")
    print("-" * 50)

if __name__ == "__main__":
    export_to_onnx()