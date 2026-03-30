import cv2
import numpy as np
from torchvision import transforms
from PIL import Image

class FramePreprocessor:
    def __init__(self):
        # 🚨 MediaPipe를 버리고, 에러가 절대 안 나는 OpenCV 기본 얼굴 인식기 사용!
        cascPath = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascPath)
        
        # ImageNet 정규화 파이프라인
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def process(self, image_bytes: bytes):
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None: return None

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # OpenCV 얼굴 인식은 흑백 이미지를 사용합니다
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) 
        
        # 얼굴 탐지 실행
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # 인식된 얼굴이 없으면 통과
        if len(faces) == 0: return None

        # 첫 번째 인식된 얼굴의 좌표 (x, y, 너비, 높이)
        x, y, w, h = faces[0]
        
        # MediaPipe 때처럼 여백(Margin)을 주어 크롭
        margin = 0.2
        img_h, img_w, _ = img_rgb.shape
        
        x_min = max(0, int(x - w * margin / 2))
        y_min = max(0, int(y - h * margin / 2))
        box_w = min(img_w - x_min, int(w * (1 + margin)))
        box_h = min(img_h - y_min, int(h * (1 + margin)))

        face_crop = img_rgb[y_min:y_min+box_h, x_min:x_min+box_w]
        if face_crop.size == 0: return None

        # 텐서 변환
        pil_img = Image.fromarray(face_crop)
        tensor = self.transform(pil_img) 
        
        return tensor.numpy()