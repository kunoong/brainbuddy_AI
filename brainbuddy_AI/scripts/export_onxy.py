```python
# scripts/export_onnx.py
import torch
import torch.nn as nn
from torchvision import models
import os

# ==========================================
# 1. 모델 구조 정의 (clip2_1.py에서 그대로 가져옴)
# ==========================================
class CNNEncoder(nn.Module):
    def __init__(self, backbone="resnet18"):
        super().__init__()
        self.out_dim = 512
        if backbone == "resnet18":
            # 사전 학습된 가중치를 다운로드하지만, LSTM 헤드는 랜덤 초기화됩니다.
            m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            self.encoder = nn.Sequential(*list(m.children())[:-1])  # (B,512,1,1)
            self.out_dim = 512
        else:
            raise ValueError("Unsupported backbone")

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
        
    def forward(self, x):  # (B,T,3,H,W)
        B,T,C,H,W = x.shape
        x = x.reshape(B*T, C, H, W)
        feats = self.cnn(x).view(B, T, -1)
        seq, _ = self.lstm(feats)
        pooled = seq.mean(dim=1)
        return self.head(pooled).squeeze(1)

# ==========================================
# 2. 서빙용 Wrapper 클래스 (Sigmoid 추가)
# ==========================================
class ServingWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # 모델에서 logit을 뽑고 Sigmoid를 거쳐 0~1 사이의 확률값으로 변환
        logits = self.model(x)
        probs = torch.sigmoid(logits)
        return probs

# ==========================================
# 3. ONNX 추출 로직
# ==========================================
def export_to_onnx():
    print("🚀 ONNX 변환을 시작합니다...")
    
    # 모델 인스턴스 생성 (원본 코드의 기본 설정값 사용)
    base_model = CNN_LSTM(backbone="resnet18", hidden=256, num_layers=2, bidirectional=True)
    
    # 서빙용 래퍼 씌우기
    model = ServingWrapper(base_model)
    
    # 필수: 추론 모드(eval)로 전환 (Dropout, BatchNorm 비활성화)
    model.eval()

    # 입력 더미 데이터 생성 (Batch=1, Seq=30, Channel=3, Height=224, Width=224)
    dummy_input = torch.randn(1, 30, 3, 224, 224)

    # 저장할 디렉토리 생성
    os.makedirs("../onnx_model", exist_ok=True)
    onnx_path = "../onnx_model/brainbuddy_random.onnx"

    # ONNX 내보내기
    torch.onnx.export(
        model,                      # 실행할 모델
        dummy_input,                # 모델에 들어갈 더미 입력값
        onnx_path,                  # 저장될 경로
        export_params=True,         # 모델 안에 학습된 가중치를 저장
        opset_version=14,           # ONNX 버전 (14가 최신 안정화 버전)
        do_constant_folding=True,   # 최적화: 상수 폴딩
        input_names=['input_frames'],       # 입력 텐서 이름 (FastAPI에서 이 이름 사용)
        output_names=['engagement_prob'],   # 출력 텐서 이름
        dynamic_axes={                      # 배치 사이즈를 가변적으로 허용
            'input_frames': {0: 'batch_size'},
            'engagement_prob': {0: 'batch_size'}
        }
    )
    
    print(f"✅ 성공! ONNX 모델이 저장되었습니다: {onnx_path}")
    print("이 모델은 이제 FastAPI 서버에서 'onnxruntime'으로 실행될 준비가 되었습니다.")

if __name__ == "__main__":
    export_to_onnx()
```
