# 프로젝트 명
BrainBuddyAI : Deep Learning Based Engagement Measuring Model (CNN → LSTM)

본 프로젝트에서는 **CNN → LSTM 구조**를 활용하여  
영상 데이터를 기반으로 **집중도**를 측정하는 모델을 구현하고,  
다양한 하이퍼파라미터 및 모델 구조 변경 실험을 통해 최적의 성능을 탐색하였습니다.
<br><br>

## 기술 스택
- 언어 & 환경
![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
	- Python 3.10.0

- 딥러닝 / 모델링
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=pytorch&logoColor=white)
![Torchvision](https://img.shields.io/badge/Torchvision-%23EE4C2C.svg?logo=pytorch&logoColor=white)
	- PyTorch – 모델 구현 및 학습
	- Torchvision – CNN 백본 및 이미지 변환

- 컴퓨터 비전 / 전처리
![OpenCV](https://img.shields.io/badge/OpenCV-%23white.svg?logo=opencv&logoColor=black)
![Mediapipe](https://img.shields.io/badge/Mediapipe-4285F4?logo=google&logoColor=white)
	- OpenCV – 영상 프레임 처리
	- Mediapipe FaceDetection – 얼굴 검출 및 크롭

- 평가 & 시각화
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E.svg?logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-013243.svg?logo=plotly&logoColor=white)
![UMAP](https://img.shields.io/badge/UMAP--learn-5D3FD3.svg?logo=python&logoColor=white)
	- scikit-learn – 평가 지표 (F1, Recall, Confusion Matrix)
	- Matplotlib – 학습 곡선, 혼동 행렬, 시각화
	- UMAP-learn – 임베딩 차원 축소 및 시각화
<br>

## 📂 폴더 구조
```bash
├── EDA
│   ├── light_color_diff.py
│   ├── t_SNE.py
│   └── umap_features.py
├── datasets
│   ├── video_folder_dataset.py
└── models
│   ├── cnn_encoder.py
│   ├── engagement_model.py
│   └── face_crop.py
└── preprocessing
│   ├── pickle_labels/
│   ├── check_label.py
│   ├── extract_frames.py
│   ├── extract_test_frames.py
│   └── labeling.py
└── test1/
└── test2/
└── test3/
└── test4/
└── real_time.py
└── test.py
└── train.py
``` 
<br>

## 0. 모델 구조
<img src="https://github.com/user-attachments/assets/4aace760-7b52-4cb1-bda2-6202143f7e62" width="500" ><br>
30프레임 시퀀스 -> CNN(MobileNetV3-Large) -> LSTM -> 집중여부(0/1)
<br><br><br>

## 1. 데이터 
### 사용 데이터셋
[학습태도 및 성향 관찰 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=71715)
<br>

### 전처리 및 라벨링
`python -m preprocess2.ext` mediapipe로 facecrop 후 10초에 30frame씩 추출

`python -m preprocess2.labeling` (폴더 경로, 라벨) 값을 .pkl 에 저장
<br><br>

## 2. 모델 학습
`python train.py`
- Epoch: 15
- Early Stopping patience: 4
- Batch size = 4
- Optimizer: AdamW
- Loss Function: BCEWithLogitsLoss + CosineAnnealingLR
- Gradient Accumulation = 8 step
<br><br>
 
## 3. 모델 테스트 및 성능
 
| test set |  Accuracy | Recall | F1 |
| --- |--- | --- | --- |
| AIhub test set | 0.8088 | 0.8690 | 0.8149 |
| 자체 개발 test set | 0.8103 | 0.8911 | 0.7993 |

<br>

모델 실험 과정에서는 AIHub의 데이터셋을, 최종 모델 평가에서는 팀원들이 직접 웹캠으로 촬영한 5분 내외의 자체 개발 test set을 사용하였습니다.
<br>
<p align="center">
  <img src="https://github.com/user-attachments/assets/f10c8132-fd12-48e8-a942-6c880c4e3ae9" width="51%">
  <img src="https://github.com/user-attachments/assets/e21fd614-98ae-430b-bffc-8d64eddc1d8f" width="47%">
</p>

<br><br>

## 4. 직접 로컬에서 실행해보기
1. "best_model.pt"를 다운로드
2. `real_time.py`의 CKPT_PATH에 해당 .pt 경로 지정
3. `python real_time.py`
<br><br>
   *T_WINDOW와 STRIDE_SEC를 바꾸어 윈도우 크기, 추론 시간을 조정할 수 있습니다.<br>
   *Defalut : 학습과정과 동일하게 T_WINDOW =30, STRIDE_SEC =5 따라서 집중도 화면의 결과값에는 대략 5초의 딜레이가 있습니다.<br>
 <br>
 <br>

