import os
import json
import pickle

label_base_dir = os.path.normpath(
    r"C:/Users/user/Downloads/109.학습태도 및 성향 관찰 데이터/3.개방데이터/1.데이터/Training/02.라벨링데이터/TL_20_03"
)
train_base_dir = os.path.normpath(r"C:/AIhub_frames2/train") # frame 폴더가 저장된 경로
output_pickle_path = os.path.normpath("../labels/train/20_03.pkl") # (폴더경로, 라벨)값을 저장할 pickle

label_texts = set()
label_map = {
    "집중": 1,
    "집중하지않음": 0,
}

results = []
file_count = 0
parsed_count = 0
skipped_count = 0

print(f"\n라벨 폴더 확인: {label_base_dir}")
if not os.path.exists(label_base_dir):
    print("❌ 경로가 존재하지 않음! 경로를 확인하세요.")
    exit()

# 전체 JSON 경로 순회
for root, _, files in os.walk(label_base_dir):
    for file in files:
        if file.endswith(".json"):
            file_count += 1
            json_path = os.path.normpath(os.path.join(root, file))
            print(f"처리 중: {json_path}")

            filename = os.path.splitext(file)[0]
            try:
                *prefix_parts, num_folder = filename.split("-")
                folder_name = "-".join(prefix_parts)
                segment_path = os.path.normpath(os.path.join(train_base_dir, folder_name, f"segment_{num_folder}"))

                # ✅ 폴더 존재 여부 확인
                if not os.path.isdir(segment_path):
                    print(f"🚫 segment 폴더 없음: {segment_path}")
                    skipped_count += 1
                    continue

                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                label_text = data["이미지"]["category"]["name"].strip()
                label = label_map[label_text]
                label_texts.add(label_text)

                results.append((segment_path, label))
                parsed_count += 1

            except KeyError:
                print(f"⚠️ 라벨 없음: {json_path}")
                skipped_count += 1
            except Exception as e:
                print(f"❌ 오류 발생: {json_path} → {e}")
                skipped_count += 1

# Pickle 저장 폴더 생성
os.makedirs(os.path.dirname(output_pickle_path), exist_ok=True)

with open(output_pickle_path, 'wb') as f:
    pickle.dump(results, f)

print("\n완료된 파일 수:", parsed_count)
print("스킵된 파일 수:", skipped_count)
print("Pickle 저장 위치:", output_pickle_path)

# 예시 출력
print("\n🎯 예시 출력 (최대 5개):")
for item in results[:5]:
    print(item)
print(f"\n총 {len(results)}개의 데이터가 저장되었습니다.")
print("\n 등장한 라벨 문자열:")
for label in sorted(label_texts):
    print(f"  '{label}'")