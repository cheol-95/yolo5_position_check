import cv2
import numpy as np
from yolov5 import YOLOv5

# 모델 경로 및 이미지 경로 설정
MODEL_PATH = './path/to/save/model.pt'
IMAGE_PATH = './path/to/image.jpg'

# YOLOv5 모델 로드
yolov5 = YOLOv5(model=MODEL_PATH, device='cpu')

# 이미지 로드
image = cv2.imread(IMAGE_PATH)

# 문틀 탐지
results = yolov5.detect(image)

# 탐지 결과 처리
for result in results:
    label = result['label']
    x, y, w, h = result['x'], result['y'], result['w'], result['h']
    center_x = x + w // 2
    
    # 정위치 판단
    if label == 'left_frame' and 120 <= center_x <= 130:
        print('좌문틀 정위치입니다!')
    elif label == 'right_frame' and 240 <= center_x <= 250:
        print('우문틀 정위치입니다!')
    else:
        print('정위치가 아닙니다.')

# 결과 이미지 표시
cv2.imshow('Detected Frames', image)
cv2.waitKey(0)
cv2.destroyAllWindows() 