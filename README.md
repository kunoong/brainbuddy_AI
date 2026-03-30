# 🧠 BrainBuddy — Real-time Engagement Analysis

> **K-SoftVation Showcase 2025 — Grand Prize** (Minister of Science and ICT Award)  
> Real-time engagement detection from webcam using Deep Learning

---

## 🏆 Award

| Item | Detail |
|------|--------|
| Competition | K-Software Empowerment BootCamp — K-SoftVation Showcase 2025 |
| Prize | **Grand Prize** (정보통신기획평가원장상) |
| Team | Queen Never Cry |
| Date | August 28, 2025 |

---

## 📌 Overview

BrainBuddy detects whether a person is **focused or unfocused** in real time using a webcam stream.  
The model analyzes 30-frame sequences and returns an engagement probability.

**My Role — AI & Data Lead:**
- Identified domain gap between AI Hub training data and real-world webcam environment
- Designed a 4-stage automated labeling pipeline using **CLIP (ViT-B/32)**
- Achieved **AUC 0.997 / F1 0.963** on CNN (ResNet18) + Bi-LSTM
- Built a complete **FastAPI + WebSocket serving pipeline** with ONNX optimization (post-competition)

---

## 📂 Project Structure

```
brainbuddy_AI/
│
├── 📁 app/                       # [추가] FastAPI 서빙 파이프라인
│   ├── main.py                   # WebSocket 서버
│   └── services/
│       ├── preprocessor.py       # BGR→RGB, 얼굴 크롭, 정규화
│       └── inference.py          # ONNX Runtime 추론 엔진
│
├── 📁 scripts/                   # [추가] 모델 변환
│   └── export_onnx.py            # PyTorch → ONNX 변환
│
├── 📁 onnx_model/                # [추가] ONNX 모델 저장
│
├── 📁 preprocessing/             # CLIP 라벨링 파이프라인
│   ├── clip1_1.py                # Zero-shot 라벨링
│   ├── clip1_2.py                # Confidence 구간 분리
│   ├── clip1_3.py                # Linear Probe 학습
│   └── clip1_4.py                # 전체 데이터 재분류
│
├── 📁 models/                    # 모델 아키텍처
│   ├── cnn_encoder.py
│   ├── engagement_model.py
│   └── face_crop.py
│
├── 📁 EDA/                       # 탐색적 데이터 분석
├── 📁 datasets/                  # 데이터셋 로더
├── 📁 test1~4/                   # 실험 로그 & 분석
│
├── real_time.py                  # 로컬 실시간 추론
├── train.py                      # 모델 학습
└── requirements.txt
```

---

## Part 1. 🔬 AI Modeling (K-SoftVation 대회)

### 문제 — 도메인 갭 (Domain Gap)

AI Hub 안구추적 데이터셋으로 학습한 모델이 실제 웹캠 환경에서 완전히 실패.  
원인: 기존 라벨의 신뢰도 문제 (고개 돌림 → "비집중", 스마트폰 시청 → "집중" 오표기)

### 해결 — CLIP 기반 라벨링 자동화 파이프라인

| 단계 | 파일 | 내용 | 결과 |
|------|------|------|------|
| Step 1 | clip1_1.py | CLIP Zero-shot 라벨링 | 1,464 폴더 처리 |
| Step 2 | clip1_2.py | Confidence 구간 분리 | high 147개 확보 |
| Step 3 | clip1_3.py | Linear Probe 학습 | ROC-AUC 0.9806 |
| Step 4 | clip1_4.py | 전체 재분류 (threshold=0.700) | 집중 890 / 비집중 574 |

### 모델 구조

```
CNN (ResNet18, ImageNet pretrained)
    ↓ frame-level feature extraction (30 frames)
Bi-LSTM (hidden=256, layers=2)
    ↓ temporal sequence analysis
FC → Sigmoid → Focused / Unfocused
```

### 학습 설정

| Parameter | Value |
|-----------|-------|
| Input | 30-frame sequence, 224×224 |
| Optimizer | Adam (lr=3e-4) |
| Loss | BCEWithLogitsLoss |
| Scheduler | CosineAnnealingLR |
| Early Stopping | patience=5, monitor=AUC |
| Validation | Group-split (data leakage 방지) |

### 최종 성능

| Metric | Score |
|--------|-------|
| AUC | **0.997** |
| F1-Score | **0.963** |
| Accuracy | 0.9505 |
| Recall | 0.9778 |

<p align="center">
  <img src="https://github.com/user-attachments/assets/f10c8132-fd12-48e8-a942-6c880c4e3ae9" width="51%">
  <img src="https://github.com/user-attachments/assets/e21fd614-98ae-430b-bffc-8d64eddc1d8f" width="47%">
</p>

---

## Part 2. ⚡ Serving Pipeline (개인 추가 작업 — 2026.03.30)

### 배경

대회 당시 백엔드 연동 시 예측 확률이 0.02로 고정되는 치명적 버그 발생.  
**원인:** 전처리 파이프라인 불일치 (BGR→RGB 변환 누락, 얼굴 크롭 미적용, ImageNet 정규화 누락)  
→ 대회 종료 후 개인 프로젝트로 완전 해결.

### 시스템 아키텍처

```
Webcam Stream
     ↓
WebSocket (real-time frame reception, 10+ FPS)
     ↓
preprocessor.py
  ├─ BGR → RGB conversion      ← 핵심 버그 수정
  ├─ Face crop (Haar Cascade)  ← 얼굴 크롭 이식
  └─ ImageNet normalization     ← 정규화 이식
     ↓
Sliding window buffer (deque, maxlen=30)
     ↓
inference.py (ONNX Runtime, CPUExecutionProvider)
     ↓
Engagement (%) + Latency (ms)
```

### 핵심 구현

**① preprocessor.py — 전처리 엔진**
- 과거 실패 원인 3단계 완벽 모듈화 (BGR→RGB, 얼굴 크롭, ImageNet 정규화)
- 트러블슈팅: Python 3.13 환경 MediaPipe 충돌 → OpenCV Haar Cascade로 전면 교체

**② export_onnx.py — ONNX 변환기**
- PyTorch 모델 → 배포용 ONNX 포맷 변환
- `ServingWrapper`: Sigmoid 연산 모델 내부에 포함
- `dynamic_axes`: 배치 사이즈 확장 유연 대응

**③ inference.py — 추론 엔진**
- `onnxruntime` CPUExecutionProvider 활용
- Latency 측정 로직 포함 → 응답 페이로드에 포함

**④ main.py — FastAPI 서버**
- WebSocket 채택 → 초당 10프레임 이상 실시간 수신
- `collections.deque(maxlen=30)` 슬라이딩 윈도우 버퍼
- 30프레임 버퍼가 찰 때만 추론 → 메모리 최적화

### 성과

| 항목 | 결과 |
|------|------|
| End-to-End 파이프라인 | ✅ 웹캠 → 집중도(%) 출력 전체 사이클 정상 구동 |
| CPU Latency | ✅ **약 890ms** (30프레임 기준, 1초 이내 실시간 처리) |
| 전처리 이식 | ✅ BGR→RGB, 얼굴 크롭, ImageNet 정규화 완벽 구현 |

---

## 🚀 How to Run

### 로컬 실시간 추론 (대회 버전)
```bash
# 1. 모델 가중치 다운로드 후 경로 지정
# real_time.py의 CKPT_PATH 수정

# 2. 실행
python real_time.py
```
> T_WINDOW=30, STRIDE_SEC=5 기준 약 5초 딜레이 발생

### FastAPI 서빙 (개인 추가 버전)
```bash
# 1. 패키지 설치
pip install -r requirements.txt

# 2. ONNX 변환
python scripts/export_onnx.py

# 3. 서버 실행
uvicorn app.main:app --reload

# 4. WebSocket 클라이언트로 ws://localhost:8000/ws 연결
```

---

## 🛣️ Roadmap

- [x] CLIP 기반 라벨링 파이프라인
- [x] CNN + Bi-LSTM 모델 학습 (AUC 0.997)
- [x] FastAPI + WebSocket 서빙 파이프라인
- [x] ONNX 변환 + Latency 최적화 (~890ms)
- [x] 개인 GitHub 레포 이관
- [ ] Docker Compose 배포 환경 구성
- [ ] 아키텍처 다이어그램 추가
- [ ] 실제 학습 가중치 확보 후 정확도 검증

---

## 🧰 Tech Stack

| Category | Tech |
|----------|------|
| Modeling | PyTorch, ResNet18, Bi-LSTM, CLIP (ViT-B/32) |
| Preprocessing | OpenCV, Haar Cascade, Torchvision, MediaPipe |
| Labeling | CLIP Zero-shot, Logistic Regression (Linear Probe) |
| Serving | FastAPI, WebSocket, ONNX Runtime |
| Evaluation | scikit-learn, UMAP, Matplotlib |

---

## 👤 My Role

**AI & Data Lead** — Team Queen Never Cry

| 기여 항목 | 내용 |
|-----------|------|
| 도메인 갭 발견 | 기존 라벨 신뢰도 문제 직접 발견 및 원인 분석 |
| CLIP 파이프라인 | 4단계 자동 라벨링 파이프라인 설계 및 구현 |
| 모델 학습 | ResNet18 + Bi-LSTM, AUC 0.997 달성 |
| 서빙 파이프라인 | FastAPI + WebSocket + ONNX 전체 구조 설계 및 구현 |

---
