import os
import cv2
from yolov5 import YOLOv5

# 이미지 디렉토리 경로
IMAGE_DIR = './cropped_images/x'
LABEL_DIR = './cropped_images/labels'  # 라벨 파일이 저장될 디렉토리

# 라벨 디렉토리가 없으면 생성
if not os.path.exists(LABEL_DIR):
    os.makedirs(LABEL_DIR)

# 이미지 파일 목록 가져오기
image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg')]

# 각 이미지 파일에 대해 처리
for image_file in image_files:
    # 이미지 경로
    image_path = os.path.join(IMAGE_DIR, image_file)
    
    # 이미지 읽기
    image = cv2.imread(image_path)
    
    # 이미지 크기 가져오기
    height, width, _ = image.shape
    
    # 라벨 파일 경로
    label_file = os.path.splitext(image_file)[0] + '.txt'
    label_path = os.path.join(LABEL_DIR, label_file)
    
    # 라벨 파일 작성 (예: YOLO 형식)
    with open(label_path, 'w') as f:
        # 예시: 클래스 0, 바운딩 박스 중심 x, y, 너비, 높이 (정규화된 값)
        # 여기서는 임의의 값으로 예시를 작성합니다.
        class_id = 0  # 예: 문틀 클래스
        x_center = 0.5  # 이미지 너비의 절반
        y_center = 0.5  # 이미지 높이의 절반
        bbox_width = 0.2  # 이미지 너비의 20%
        bbox_height = 0.2  # 이미지 높이의 20%
        
        # YOLO 형식으로 라벨 작성
        f.write(f"{class_id} {x_center} {y_center} {bbox_width} {bbox_height}\n")

print("데이터셋 파싱 완료!")

# 데이터셋 경로 설정
DATASET_PATH = './cropped_images'
MODEL_SAVE_PATH = './models/yolov5s_train_cropped.pt'

# YOLOv5 모델 초기화
yolov5 = YOLOv5(model='yolov5s', device='cpu')

# 데이터셋 로드
yolov5.load_dataset(DATASET_PATH)

# 모델 학습
yolov5.train(epochs=50)

# 학습된 모델 저장
yolov5.save(MODEL_SAVE_PATH)

print('모델 학습 완료 및 저장됨:', MODEL_SAVE_PATH) 