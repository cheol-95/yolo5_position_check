import cv2
import os

# 설정
IMAGE_DIR = './2_position_sample_train_labeled_frames'
OUTPUT_DIR = './cropped_images'

# 이미지 자르기 함수
def crop_image(image_path, output_path):
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    
    # 중간 2/4 부분만 남기기 (좌우)
    sixth_width = width // 6
    cropped_image = image[:, sixth_width:5*sixth_width]
    
    # 상하 6분할하여 중간 2/6 부분만 남기기
    eighth_height = height // 8
    final_cropped_image = cropped_image[4*eighth_height:6*eighth_height, :]
    
    # 자른 이미지를 저장
    cv2.imwrite(output_path, final_cropped_image)

# 디렉토리 내 모든 이미지 처리
for category in os.listdir(IMAGE_DIR):
    category_path = os.path.join(IMAGE_DIR, category)
    if not os.path.isdir(category_path):
        continue
    
    # 출력 디렉토리 구조 생성
    output_category_path = os.path.join(OUTPUT_DIR, category)
    os.makedirs(output_category_path, exist_ok=True)
    
    for image_name in os.listdir(category_path):
        image_path = os.path.join(category_path, image_name)
        output_path = os.path.join(output_category_path, image_name)
        crop_image(image_path, output_path)
        print(f"✅ 저장됨: {output_path}")