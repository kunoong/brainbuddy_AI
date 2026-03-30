import cv2
import os
import shutil
import pickle
from tqdm import tqdm
from models.face_crop import crop_face
import mediapipe as mp

def extract_frames(video_path, local_output_base, face_detector,
                   segment_duration=10, target_fps=3, max_frames=30):
    """
    주어진 비디오를 segment_duration(초) 단위로 나누어 각 세그먼트에서
    target_fps로 얼굴을 크롭/저장하여 세그먼트당 최대 max_frames장을 만듭니다.
    세그먼트 폴더는 local_output_base/segment_{idx}로 저장됩니다.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"영상 열기 실패 : {video_path}")
        return

    fps = 30  # 영상 원본 FPS 가정(필요시 cap.get(cv2.CAP_PROP_FPS)로 대체 가능)
    frame_interval = max(int(fps / target_fps), 1)
    segment_frame_count = segment_duration * fps
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        print(f"⚠️ 프레임 수를 읽지 못함: {video_path}")
        cap.release()
        return
    num_segments = total_frames // segment_frame_count

    print(f"🎬 세그먼트 수: {num_segments}, Interval: {frame_interval}프레임마다 저장")

    for segment_idx in tqdm(range(num_segments), desc="30프레임 단위로 분리"):
        local_segment_dir = os.path.normpath(os.path.join(local_output_base, f"segment_{segment_idx}"))

        if os.path.exists(local_segment_dir):
            jpg_files = [f for f in os.listdir(local_segment_dir) if f.lower().endswith(".jpg")]
            if len(jpg_files) >= max_frames:
                # 이미 충분히 있음 → 건너뜀
                # print(f"✅ 세그먼트 {segment_idx} 이미 {len(jpg_files)}장 존재 → 건너뜀.")
                continue
            else:  # 30장이 아니면 지우고 덮어씀
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

    cap.release()
    print("전체 작업 완료")


def find_valid_segments(base_dir, min_frames=30):
    """
    base_dir 하위의 segment_* 폴더 중 JPG가 min_frames장 이상 있는 폴더 경로 리스트 반환
    """
    valid = []
    if not os.path.isdir(base_dir):
        return valid
    for name in os.listdir(base_dir):
        if name.startswith("segment_"):
            seg_dir = os.path.normpath(os.path.join(base_dir, name))
            if os.path.isdir(seg_dir):
                jpg_files = [f for f in os.listdir(seg_dir) if f.lower().endswith(".jpg")]
                if len(jpg_files) >= min_frames:
                    valid.append(seg_dir)
    return sorted(valid)


if __name__ == "__main__":
    # ====== 입력/출력 루트 설정 ======
    # 입력 영상 루트: C:\our_data\1 (라벨 1), C:\our_data\0 (라벨 0)
    input_roots = {
        1: r"C:/our_data/1",
        0: r"C:/our_data/0",
    }
    # 프레임 저장 루트(원하는 경로로 변경 가능)
    output_root = r"C:/our_frames"
    os.makedirs(output_root, exist_ok=True)

    # 저장할 pkl 경로
    pkl_path = "C:/KSEB/brainbuddy_AI/preprocess2/pickle_labels/test/our_dataset.pkl"

    # ====== 얼굴 검출기 초기화 ======
    mp_face_detection = mp.solutions.face_detection
    face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

    # ====== (폴더 경로, 라벨) 데이터셋 쌓기 ======
    dataset = []  # (segment_dir_path, label)

    for label, video_folder in input_roots.items():
        if not os.path.isdir(video_folder):
            print(f"⚠️ 입력 폴더가 없습니다: {video_folder}")
            continue

        video_files = sorted([f for f in os.listdir(video_folder) if f.lower().endswith((".mp4", ".mov"))])

        print(f"\n===== 라벨 {label} | 폴더: {video_folder} | 영상 수: {len(video_files)} =====")
        for video_file in video_files:
            video_path = os.path.join(video_folder, video_file)
            video_name = os.path.splitext(video_file)[0]
            local_output_base = os.path.join(output_root, str(label), video_name)
            os.makedirs(local_output_base, exist_ok=True)

            # 프레임 추출
            extract_frames(video_path, local_output_base, face_detector,
                           segment_duration=10, target_fps=3, max_frames=30)

            # 이번 비디오에서 유효한 세그먼트(30장 이상)만 수집
            valid_segments = find_valid_segments(local_output_base, min_frames=30)
            for seg_dir in valid_segments:
                # (세그먼트 폴더 경로, 라벨) 튜플 저장
                dataset.append((os.path.normpath(seg_dir), label))

        print(f"라벨 {label} 처리 완료. 누적 세그먼트 수: {len(dataset)}")

    # 얼굴 검출기 종료
    face_detector.close()

    # ====== PKL 저장 ======
    with open(pkl_path, "wb") as f:
        pickle.dump(dataset, f)

    print("\n================ 저장 요약 ================")
    print(f"총 세그먼트(30프레임 폴더) 수: {len(dataset)}")
    print(f"PKL 저장 위치: {os.path.normpath(pkl_path)}")
    if len(dataset) > 0:
        print("샘플 3개:")
        for sample in dataset[:3]:
            print(sample)
