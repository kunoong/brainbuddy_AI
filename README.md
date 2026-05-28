# 🧠 BrainBuddy — Real-time Human Focus Analytics System

> **K-SoftVation Showcase 2025 — Grand Prize** (Minister of Science and ICT Award)
> Real-time human focus analytics system with optimized AI inference serving and WebSocket-based streaming architecture.

---

# 🏆 Award

| Item        | Detail                                                       |
| ----------- | ------------------------------------------------------------ |
| Competition | K-Software Empowerment BootCamp — K-SoftVation Showcase 2025 |
| Prize       | **Grand Prize** (정보통신기획평가원장상)                                |
| Team        | Queen Never Cry                                              |
| Date        | August 28, 2025                                              |

---

# 📌 Overview

BrainBuddy is a real-time human focus analytics system that estimates user engagement levels from webcam streams.

The project was designed to address the gap between high offline model performance and unstable real-world webcam inference environments.

The system combines:

* Computer Vision
* Real-time inference serving
* ONNX optimization
* WebSocket streaming
* Data-centric AI pipeline design

---

# 🚀 Key Highlights

| Metric            | Result                             |
| ----------------- | ---------------------------------- |
| AUC               | **0.997**                          |
| F1-Score          | **0.963**                          |
| Accuracy          | 0.9505                             |
| Recall            | 0.9778                             |
| CPU Latency       | **~890ms**                         |
| Serving Framework | FastAPI + WebSocket + ONNX Runtime |

---

# 🧠 My Role — AI & Data Lead

* Identified domain gap between AI Hub training data and real-world webcam environments
* Designed a 4-stage automated labeling pipeline using CLIP (ViT-B/32)
* Trained CNN (ResNet18) + Bi-LSTM sequence model
* Built real-time FastAPI + WebSocket serving pipeline
* Solved preprocessing mismatch issues causing unstable production inference
* Optimized deployment using ONNX Runtime

---

# 🏗️ System Architecture

```text
Webcam Stream
      ↓
WebSocket Streaming
      ↓
Preprocessing Pipeline
  ├─ BGR → RGB Conversion
  ├─ Face Cropping
  └─ ImageNet Normalization
      ↓
Sliding Window Buffer (30 Frames)
      ↓
ONNX Runtime Inference
      ↓
Focus Score (%) + Latency Output
```

---

# 📂 Project Structure

```bash
brainbuddy_AI/
│
├── app/
│   ├── main.py
│   └── services/
│       ├── preprocessor.py
│       └── inference.py
│
├── scripts/
│   └── export_onnx.py
│
├── onnx_model/
│
├── preprocessing/
│   ├── clip1_1.py
│   ├── clip1_2.py
│   ├── clip1_3.py
│   └── clip1_4.py
│
├── models/
├── datasets/
├── EDA/
│
├── train.py
├── real_time.py
└── requirements.txt
```

---

# 🔬 AI Modeling Pipeline

## Problem — Domain Gap

Models trained on the AI Hub eye-tracking dataset failed completely in real webcam environments.

Root causes identified:

* unreliable labels
* noisy engagement annotations
* mismatch between training and inference distributions

Examples:

* head-turning labeled as “unfocused”
* smartphone viewing labeled as “focused”

---

## Solution — CLIP-based Automated Labeling Pipeline

| Step   | Description                   | Result                      |
| ------ | ----------------------------- | --------------------------- |
| Step 1 | CLIP zero-shot labeling       | 1,464 folders processed     |
| Step 2 | Confidence interval filtering | 147 high-confidence samples |
| Step 3 | Linear Probe training         | ROC-AUC 0.9806              |
| Step 4 | Full dataset relabeling       | Focused 890 / Unfocused 574 |

---

# 🧩 Model Architecture

```text
CNN (ResNet18, ImageNet pretrained)
        ↓
Frame-level feature extraction
        ↓
Bi-LSTM sequence modeling
        ↓
Fully Connected Layer
        ↓
Sigmoid → Focused / Unfocused
```

---

# ⚡ Real-time Serving Pipeline

## Engineering Challenge

During the original competition integration phase, the backend inference pipeline consistently returned:

```python
prob ≈ 0.02
```

As a result, the model always predicted the “unfocused” state regardless of input.

Due to time constraints, another model was used for the final competition submission.

After the competition, I independently rebuilt the serving pipeline and identified the root causes.

---

## Root Cause Analysis

The issue was caused by preprocessing inconsistency between training and inference environments.

### Key issues identified

* Missing BGR → RGB conversion
* Missing face cropping pipeline
* Missing ImageNet normalization

These mismatches caused severe inference distribution shift.

---

# 🔧 Serving Pipeline Reconstruction

## ① Preprocessing Engine (`preprocessor.py`)

Implemented:

* BGR → RGB conversion
* face cropping
* ImageNet normalization

Additional troubleshooting:

* MediaPipe compatibility issues in Python 3.13
* Replaced with OpenCV Haar Cascade pipeline

---

## ② ONNX Optimization (`export_onnx.py`)

* Converted PyTorch model to ONNX Runtime
* Embedded sigmoid operation into `ServingWrapper`
* Configured dynamic batch axes

---

## ③ Inference Engine (`inference.py`)

* ONNX Runtime CPUExecutionProvider
* Real-time latency measurement logic

### Result

* Achieved approximately **890ms latency**
* Measured using 30-frame sliding window inference

---

## ④ Real-time Streaming Server (`main.py`)

* FastAPI + WebSocket architecture
* 10+ FPS real-time frame streaming
* Sliding window buffering:

```python
collections.deque(maxlen=30)
```

### Benefits

* memory optimization
* stable sequence inference
* reduced redundant processing

---

# 📊 Performance Evaluation

| Metric                | Score  |
| --------------------- | ------ |
| AUC                   | 0.997  |
| F1-Score              | 0.963  |
| Accuracy              | 0.9505 |
| Recall                | 0.9778 |
| CPU Inference Latency | ~890ms |

---

# 🐳 Deployment

## Run FastAPI Server

```bash
pip install -r requirements.txt
python scripts/export_onnx.py
uvicorn app.main:app --reload
```

Connect WebSocket client to:

```text
ws://localhost:8000/ws
```

---

# 🛣️ Roadmap

* [x] CLIP-based labeling pipeline
* [x] CNN + Bi-LSTM training pipeline
* [x] FastAPI + WebSocket real-time serving
* [x] ONNX optimization
* [x] Real-time latency optimization
* [x] Personal repository migration
* [ ] Docker Compose deployment
* [ ] Monitoring & logging
* [ ] Architecture visualization
* [ ] Production benchmark testing

---

# 🧰 Tech Stack

## Backend & Infrastructure

FastAPI · WebSocket · ONNX Runtime · Docker

## AI / Modeling

PyTorch · ResNet18 · Bi-LSTM · CLIP (ViT-B/32)

## Data & Analytics

scikit-learn · UMAP · NumPy · Pandas

## Computer Vision

OpenCV · Haar Cascade · Torchvision

---

# 📚 Lessons Learned

This project taught me that reliable AI systems depend not only on model accuracy, but also on:

* preprocessing consistency
* inference environment alignment
* deployment stability
* real-time serving architecture
* production debugging capability
* latency optimization

---

# 👤 Author

### Kunoong

Applied AI Engineer focused on:

* real-time AI systems
* AI infrastructure
* human-centered analytics
* deployable ML serving systems
