import os
import json
#20_02 집중
#20_04 집중하지 않음
#20_05 졸음
label_base_dir = r"C:/Users/user/Downloads/109.학습태도 및 성향 관찰 데이터/3.개방데이터/1.데이터/Validation/02.라벨링데이터/VL_20_02"
# label_base_dir = r"../TL_20_01"
label_text_set = set()

for root, _, files in os.walk(label_base_dir):
    for file in files:
        if file.endswith(".json"):
            try:
                with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                    data = json.load(f)
                    label_text = data["이미지"]["category"]["name"].strip()
                    label_text_set.add(label_text)
            except Exception as e:
                print(f"❌ {file}: {e}")
                continue

print("\n📋 JSON에서 발견된 라벨 종류:")
for label in sorted(label_text_set):
    print(f"  '{label}'")
