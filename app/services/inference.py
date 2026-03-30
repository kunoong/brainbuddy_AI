import onnxruntime as ort
import numpy as np
import time

class ONNXEngine:
    def __init__(self, model_path: str):
        # ONNX Runtime 세션 시작 
        # (현재는 로컬 테스트를 위해 CPU를 사용하도록 세팅했습니다)
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        
        # 우리가 export_onnx.py에서 지정했던 입력 텐서 이름('input_frames')을 자동으로 가져옵니다.
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, frame_sequence: np.ndarray):
        """
        frame_sequence: 30장의 프레임이 모인 배열 
        형태(Shape)는 [1, 30, 3, 224, 224] (Batch, Seq, Channel, Height, Width) 여야 합니다.
        """
        start_time = time.time()
        
        # 🚀 ONNX 모델에 데이터 넣고 추론 실행!
        # 여기서 원본 PyTorch보다 훨씬 빠른 속도로 계산이 일어납니다.
        outputs = self.session.run(None, {self.input_name: frame_sequence.astype(np.float32)})
        
        # 추론에 걸린 시간(Latency) 계산 (ms 단위) - 나중에 이력서에 쓸 수치!
        latency_ms = (time.time() - start_time) * 1000 
        
        # 서빙용 Wrapper에서 Sigmoid를 씌워놨기 때문에, 0.0 ~ 1.0 사이의 확률값이 나옵니다.
        prob = float(outputs[0][0]) 
        
        return prob, latency_ms