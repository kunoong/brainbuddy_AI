# realtime_focus_3fps.py
import time
import cv2
from collections import deque
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from torchvision.transforms import InterpolationMode
from PIL import Image
import mediapipe as mp
from models.face_crop import crop_face, last_face_bbox
from models.engagement_model import EngagementModel
from models.cnn_encoder import CNNEncoder
# ===== 사용자 편집 지점 =====
CKPT_PATH = "./log/train4/best_model/best_model_epoch_4.pt"  # <-- 실제 경로로 변경
CAM_INDEX = 0                                       # 기본 웹캠
TARGET_FPS = 3                                      # 3 FPS 고정
IMG_SIZE = 224
T_WINDOW = 15                                       # 학습과 동일 프레임 길이
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STRIDE_SEC = 1
STRIDE_FRAMES = max(1, int(round(TARGET_FPS * STRIDE_SEC)))  # 15

# -------- 전처리 (학습과 동일) --------
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=InterpolationMode.BILINEAR, antialias=True),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# -------- 체크포인트 로드 --------
def load_checkpoint_and_models(ckpt_path: str):
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"체크포인트를 찾을 수 없습니다: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=DEVICE)

    thr_acc = ckpt.get("thr_acc", None)
    thr_rec = ckpt.get("thr_rec", None)

    cnn = CNNEncoder().to(DEVICE)
    model = EngagementModel().to(DEVICE)
    cnn.load_state_dict(ckpt["cnn_state_dict"])
    model.load_state_dict(ckpt["model_state_dict"])
    cnn.eval(); model.eval()
    return cnn, model, thr_acc, thr_rec

@torch.no_grad()
def infer_clip(cnn, model, frames_tensor_list, threshold: float = 0.5):
    """
    frames_tensor_list: 길이 T의 list, 각 원소는 preprocess 완료된 Tensor (3,H,W) [RGB기준]
    """
    x = torch.stack(frames_tensor_list, dim=0)       # (T,3,H,W)
    x = x.unsqueeze(0).to(DEVICE)                    # (1,T,3,H,W)
    feats = cnn(x)                                   # (1,T,512)
    logit = model(feats)                             # (1,1)
    p = torch.sigmoid(logit)[0, 0].item()
    label = "focused" if p >= threshold else "unfocused"
    conf = p if label == "focused" else 1 - p
    return label, conf, p

def main():
    # --- 모델/임계값 로드 ---
    cnn, model, thr_acc, thr_rec = load_checkpoint_and_models(CKPT_PATH)
    threshold = float(thr_acc) if thr_acc is not None else (float(thr_rec) if thr_rec is not None else 0.5)
    print(f"[INFO] threshold = {threshold:.3f}")

    # --- 장치/얼굴 검출기 ---
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    face_detector = mp.solutions.face_detection.FaceDetection(
        model_selection=1,               # 0: 근거리, 1: 일반(멀리도 포함)
        min_detection_confidence=0.5
    )

    buffer = deque(maxlen=T_WINDOW)
    print("실시간 추론 시작 (3 FPS). 종료: q 키")
    interval = 1.0 / TARGET_FPS
    next_t = time.time()

    last_p = None   # <<< 추가: 최근 p 저장

    # 3 FPS 타이밍 제어
    interval = 1.0 / TARGET_FPS
    next_t = time.time()

    # 슬라이딩 윈도우 추론: 5초(15프레임)마다 갱신
    frame_idx = 0
    last_label, last_conf, last_p = "preparing", 0.0, None

    try:
        while True:
            # 타이밍 동기화 (초당 3프레임)
            now = time.time()
            if now < next_t:
                time.sleep(next_t - now)
            next_t += interval

            ok, frame_bgr = cap.read()
            if not ok:
                print("프레임 획득 실패")
                break

            # 얼굴 crop (RGB 반환) - 당신이 import한 crop_face 사용
            face_rgb = crop_face(frame_bgr, face_detector, fallback_to_full=True)
            if face_rgb is None:
                # 아무것도 못 얻으면 표시만 하고 다음 루프
                cv2.putText(frame_bgr, "No face", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow("Engagement (Real-time)", frame_bgr)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            # 전처리 → 버퍼 적재
            pil = Image.fromarray(face_rgb)
            tensor_3hw = preprocess(pil)  # (3,224,224)
            buffer.append(tensor_3hw)

            frame_idx += 1

            # 추론 타이밍: 버퍼가 가득(30프레임)이고, 15프레임(≈5초)마다
            if len(buffer) == T_WINDOW and (frame_idx % STRIDE_FRAMES == 0):
                try:
                    label, conf, p = infer_clip(cnn, model, list(buffer), threshold)
                    last_label, last_conf, last_p = label, conf, p
                except Exception as e:
                    print("Inference error:", e)
                    label, conf, p = last_label, last_conf, last_p
            else:
                # 갱신 타이밍이 아니면 직전 결과 유지
                label, conf, p = last_label, last_conf, last_p

            color = (0, 200, 0) if label == "focused" else (0, 0, 255) if label == "unfocused" else (180, 180, 0)

            # 현재 라벨에 맞는 확률만 표시
            if p is not None:
                conf_to_show = p if label in ("focused") else (1 - p if label in ("unfocused") else None)
            else:
                conf_to_show = None

            if conf_to_show is not None:
                text = f"{label}  {conf_to_show*100:.1f}%"
            else:
                text = f"{label}"

            # 얼굴 박스 그리기 (가장 최근 bbox)
            if last_face_bbox is not None:
                x1, y1, x2, y2 = last_face_bbox
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)

            # 화면 테두리 그리기
            h, w = frame_bgr.shape[:2]
            cv2.rectangle(frame_bgr, (2, 2), (w - 3, h - 3), color, 4)

            cv2.putText(frame_bgr, text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.imshow("Engagement (Real-time)", frame_bgr)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        face_detector.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
