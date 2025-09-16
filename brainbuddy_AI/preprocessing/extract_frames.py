# 영상을 가져와 30프레임씩 한 폴더에 넣어 저장하기
import cv2
import os
import shutil
from tqdm import tqdm
from models.face_crop import crop_face
import mediapipe as mp

def extract_frames(video_path, local_output_base, face_detector, segment_duration=10, target_fps=3, max_frames=30):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"영상 열기 실패 :{video_path}")
        return

    fps = 30
    frame_interval = max(int(fps / target_fps), 1)
    segment_frame_count = segment_duration * fps
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    num_segments = total_frames // segment_frame_count

    print(f"🎬 세그먼트 수: {num_segments}, Interval: {frame_interval}프레임마다 저장")

    for segment_idx in tqdm(range(num_segments), desc="30프레임 단위로 분리"):
        local_segment_dir = os.path.normpath(os.path.join(local_output_base, f"segment_{segment_idx}"))

        if os.path.exists(local_segment_dir):
            jpg_files = [f for f in os.listdir(local_segment_dir) if f.lower().endswith(".jpg")]
            if len(jpg_files) >= max_frames:
                print(f"✅ 세그먼트 {segment_idx} 이미 {len(jpg_files)}장 존재 → 건너뜀.")
                continue
            else: #30장이 아니면 지우고 덮어씀
                print(f"♻️ 세그먼트 {segment_idx} 프레임 {len(jpg_files)}장 → 덮어쓰기 위해 삭제 후 재처리")
                shutil.rmtree(local_segment_dir)

        cap.set(cv2.CAP_PROP_POS_FRAMES, segment_idx * segment_frame_count)

        count = 0
        saved = 0
        retry_count = 0

        os.makedirs(local_segment_dir, exist_ok=True)

        while saved < max_frames:
            current_frame_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
            if current_frame_pos >= (segment_idx + 1) * segment_frame_count:
                break

            ret, frame = cap.read()
            if not ret:
                if retry_count < 2:
                    retry_count += 1
                    print(f"⚠️ 세그먼트 {segment_idx} 프레임 읽기 실패, 재시도 {retry_count}/2... (프레임: {current_frame_pos})")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_pos)
                    continue
                else:
                    print(f"❌ 세그먼트 {segment_idx} 프레임 읽기 실패 초과. 건너뜀.")
                    break

            retry_count = 0

            if count % frame_interval == 0:
                cropped = crop_face(frame, face_detector)
                if cropped is not None:
                    frame_path = os.path.normpath(os.path.join(local_segment_dir, f"{saved:04d}.jpg"))
                    cropped_bgr = cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR)
                    success = cv2.imwrite(frame_path, cropped_bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    if success:
                        saved += 1
            count += 1

        #print(f"✅ 세그먼트 {segment_idx} 저장 완료 ({saved}장)")

    cap.release()
    print("전체 작업 완료")


if __name__ == "__main__":
    mp_face_detection = mp.solutions.face_detection
    face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    for i in range(1,11):# 영상 4~5개를 한 폴더에 넣어 터미널 여러개 병렬로 돌려 처리
        video_folder = f"C:/Users/user/Downloads/109.학습태도 및 성향 관찰 데이터/3.개방데이터/1.데이터/Training/01.원천데이터/TS_20_01_1/{i}"
        local_root = r"C:/AIhub_frames/train"  # ✅ 로컬 저장 위치

        video_files = sorted([
            f for f in os.listdir(video_folder)
            if f.lower().endswith(".mp4")
        ])

        for video_file in video_files:
            video_path = os.path.join(video_folder, video_file)
            video_name = os.path.splitext(video_file)[0]
            local_output_base = os.path.join(local_root, video_name)

            extract_frames(video_path, local_output_base, face_detector)
        print(f"================={i}번 폴더 전처리 완료 ===================\n")

    face_detector.close()