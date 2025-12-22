import os
import json
import time
from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.training.models import Region, ImageFileCreateEntry, ImageFileCreateBatch
from msrest.authentication import ApiKeyCredentials

# 1. Azure 리소스 설정
ENDPOINT = "https://pillclassfication.cognitiveservices.azure.com/"
TRAINING_KEY = "6T2q6i53g7IKD6yHYCk7U6uaJT0shorb4Ki55WIOrm6QPDYhIkZvJQQJ99BLACL93NaXJ3w3AAAJACOGE6xn"
PROJECT_ID = "aafa7eeb-a9f7-43ef-8d15-c6af7792f641"

# 2. 경로 설정 및 폴더 자동 탐색
current_dir = os.getcwd()
all_folders = [f for f in os.listdir(current_dir) if os.path.isdir(f)]
JSON_ROOT = next((f for f in all_folders if '라벨링' in f), "라벨링데이터")
IMAGE_ROOT = next((f for f in all_folders if '원천' in f), "원천데이터")

print(f"📂 현재 위치: {current_dir}")
print(f"✅ 인식된 라벨링 폴더: {JSON_ROOT}")
print(f"✅ 인식된 원천 폴더: {IMAGE_ROOT}")

# 모든 이미지 파일 미리 스캔 (확장자 소문자 대응)
image_map = {}
for root, dirs, files in os.walk(IMAGE_ROOT):
    for f in files:
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_map[f.lower()] = os.path.join(root, f)

print(f"검색된 이미지 파일 수: {len(image_map)}개")

# 3. Azure 클라이언트 연결 및 태그 정보 동기화
credentials = ApiKeyCredentials(in_headers={"Training-key": TRAINING_KEY})
trainer = CustomVisionTrainingClient(ENDPOINT, credentials)
tags = {t.name.strip(): t.id for t in trainer.get_tags(PROJECT_ID)}

def get_tag_id(name):
    name = name.strip()
    if name not in tags:
        print(f"🆕 새 태그 생성: {name}")
        new_tag = trainer.create_tag(PROJECT_ID, name)
        tags[name] = new_tag.id
    return tags[name]

# 4. 업로드 및 자동 박싱(Boxing) 시작
print("태깅 업로드를 시작...")
image_batch = []
total_count = 0

for root, dirs, files in os.walk(JSON_ROOT):
    for file in files:
        if file.lower().endswith('.json'):
            json_path = os.path.join(root, file)
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 데이터 유효성 검사 (unpack 에러 방지)
                if not data.get('images') or not data.get('annotations'):
                    continue
                
                img_info = data['images'][0]
                img_name = img_info['file_name'].replace('.jpg', '.png').replace('.JPG', '.png').lower()
                real_img_path = image_map.get(img_name)
                
                if real_img_path:
                    t_id = get_tag_id(img_info['dl_name'])
                    # 픽셀 좌표를 비율로 변환하기 위해 이미지 크기 로드
                    w_img, h_img = float(img_info['width']), float(img_info['height'])
                    
                    regions = []
                    for ann in data['annotations']:
                        bbox = ann.get('bbox')
                        if not bbox or len(bbox) != 4: continue
                        
                        # Azure가 요구하는 0.0 ~ 1.0 비율로 정밀 변환
                        # 좌표가 1.0을 넘지 않도록 Clamp(고정) 처리
                        left = max(0.001, min(0.99, bbox[0] / w_img))
                        top = max(0.001, min(0.99, bbox[1] / h_img))
                        width = max(0.01, min(1.0 - left, bbox[2] / w_img))
                        height = max(0.01, min(1.0 - top, bbox[3] / h_img))

                        regions.append(Region(tag_id=t_id, left=left, top=top, width=width, height=height))

                    if regions:
                        with open(real_img_path, "rb") as f_img:
                            image_batch.append(ImageFileCreateEntry(
                                name=img_name,
                                contents=f_img.read(),
                                tag_ids=[t_id],
                                regions=regions
                            ))

                    # 10장씩 묶어서 배치 업로드 (속도 향상 및 오류 방지)
                    if len(image_batch) >= 10:
                        trainer.create_images_from_files(PROJECT_ID, batch=ImageFileCreateBatch(images=image_batch))
                        total_count += len(image_batch)
                        print(f"✅ {total_count}개 업로드 및 자동 박스 생성 완료")
                        image_batch = []
                        time.sleep(0.1)

            except Exception as e:
                # Conflict는 이미 파일이 있다는 뜻이므로 무시
                if "Conflict" not in str(e):
                    print(f"❌ {file} 처리 중 오류: {e}")

# 남은 이미지 처리
if image_batch:
    trainer.create_images_from_files(PROJECT_ID, batch=ImageFileCreateBatch(images=image_batch))
    total_count += len(image_batch)

print(f"{total_count}개의 이미지가 'Tagged' 탭으로 들어감.")