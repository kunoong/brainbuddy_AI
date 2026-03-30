from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import asyncio
from collections import deque
import numpy as np
import os

# 우리가 만든 부품들 가져오기
from app.services.preprocessor import FramePreprocessor
from app.services.inference import ONNXEngine

app = FastAPI(title="BrainBuddy Real-time API")

# 부품 초기화 (서버가 켜질 때 한 번만 로드됨)
preprocessor = FramePreprocessor()

# 모델 경로 설정 (루트 디렉토리 기준)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
model_path = os.path.join(project_root, "onnx_model", "brainbuddy_random.onnx")

engine = ONNXEngine(model_path)

# ==========================================
# 1. 웹캠 테스트용 화면 (HTML) 제공
# ==========================================
html = """
<!DOCTYPE html>
<html>
    <head>
        <title>BrainBuddy WebCam Test</title>
        <style>
            body { font-family: sans-serif; text-align: center; background-color: #f4f4f9; padding: 20px;}
            video, canvas { border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
            #result { font-size: 24px; font-weight: bold; margin-top: 20px; color: #333; }
            .engaged { color: #2ecc71; }
            .unengaged { color: #e74c3c; }
        </style>
    </head>
    <body>
        <h1>🧠 BrainBuddy Real-time Test</h1>
        <video id="video" width="640" height="480" autoplay></video>
        <canvas id="canvas" width="640" height="480" style="display:none;"></canvas>
        <div id="result">Waiting for model... (Need 30 frames)</div>

        <script>
            const video = document.getElementById('video');
            const canvas = document.getElementById('canvas');
            const context = canvas.getContext('2d');
            const resultDiv = document.getElementById('result');

            // WebSocket 연결
            const ws = new WebSocket("ws://localhost:8000/ws/analyze");

            // 1. 웹캠 켜기
            navigator.mediaDevices.getUserMedia({ video: true })
                .then(stream => { video.srcObject = stream; })
                .catch(err => { console.error("Webcam error:", err); });

            // 2. 서버에서 결과를 받으면 화면에 표시
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                if (data.status === "success") {
                    const prob = (data.engagement_probability * 100).toFixed(1);
                    const isEngaged = data.is_engaged;
                    const latency = data.latency_ms;
                    
                    if (isEngaged) {
                        resultDiv.innerHTML = `<span class="engaged">집중 중 (🟢 ${prob}%)</span> <br> <small>Latency: ${latency}ms</small>`;
                    } else {
                        resultDiv.innerHTML = `<span class="unengaged">비집중 (🔴 ${prob}%)</span> <br> <small>Latency: ${latency}ms</small>`;
                    }
                } else if (data.status === "buffering") {
                    resultDiv.innerHTML = `버퍼링 중... (${data.current_frames}/30)`;
                } else if (data.error) {
                    resultDiv.innerHTML = `⚠️ ${data.error}`;
                }
            };

            // 3. 100ms(1초에 10장)마다 프레임을 서버로 전송
            setInterval(() => {
                if (ws.readyState === WebSocket.OPEN) {
                    context.drawImage(video, 0, 0, canvas.width, canvas.height);
                    canvas.toBlob(blob => {
                        if (blob) ws.send(blob);
                    }, 'image/jpeg', 0.8);
                }
            }, 100);
        </script>
    </body>
</html>
"""

@app.get("/")
async def get():
    return HTMLResponse(html)

# ==========================================
# 2. 실시간 분석 (WebSocket) 통신소
# ==========================================
@app.websocket("/ws/analyze")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # LSTM 모델은 30장의 연속된 프레임이 필요함
    frame_buffer = deque(maxlen=30) 
    
    try:
        while True:
            # 1. 클라이언트(웹캠)에서 이미지 바이트 수신
            data = await websocket.receive_bytes()
            
            # 2. 전처리 (MediaPipe 크롭 + RGB 변환 + 정규화)
            processed_frame = preprocessor.process(data)
            
            if processed_frame is None:
                await websocket.send_json({"error": "얼굴이 화면에 없습니다 (비집중)"})
                continue
                
            # 3. 버퍼에 프레임 추가
            frame_buffer.append(processed_frame)
            
            # 4. 버퍼가 30장 다 안 찼으면 기다림
            if len(frame_buffer) < 30:
                await websocket.send_json({"status": "buffering", "current_frames": len(frame_buffer)})
                continue
                
            # 5. 버퍼가 30장 꽉 찼으면 ONNX 추론 실행!
            # [30, 3, 224, 224] 형태를 -> [1, 30, 3, 224, 224] (배치 사이즈 1 추가)
            sequence_input = np.expand_dims(np.array(frame_buffer), axis=0).astype(np.float32)
            
            prob, latency = engine.predict(sequence_input)
            
            # 6. 결과 반환 (이전 1학기 때 threshold였던 0.700 기준)
            await websocket.send_json({
                "status": "success",
                "engagement_probability": prob,
                "is_engaged": prob >= 0.700,
                "latency_ms": round(latency, 2)
            })
            
    except WebSocketDisconnect:
        print("클라이언트 연결 종료")