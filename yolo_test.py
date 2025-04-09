import cv2
import numpy as np
import os
import glob

# 모델 경로 및 이미지 경로
MODEL_PATH = "models/best.onnx"
IMAGE_DIR = "sample/"
IMG_SIZE = 640
CONF_THRESHOLD = 0.25
NMS_THRESHOLD = 0.4
CLASS_NAMES = ['door']  # 학습한 클래스 이름

# 모델 불러오기
net = cv2.dnn.readNetFromONNX(MODEL_PATH)

# 이미지 리스트 불러오기
image_paths = glob.glob(os.path.join(IMAGE_DIR, "*.jpg")) + glob.glob(os.path.join(IMAGE_DIR, "*.png"))

for image_path in sorted(image_paths):
    print(f"[INFO] Processing: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print(f"[WARN] Unable to read image: {image_path}")
        continue

    h, w = img.shape[:2]

    # 이미지 전처리
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (IMG_SIZE, IMG_SIZE), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward()[0]  # (25200, 6)

    # 결과 파싱
    boxes, confidences, class_ids = [], [], []
    for det in outputs:
        scores = det[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > CONF_THRESHOLD:
            cx, cy, bw, bh = det[0:4]
            x = int((cx - bw / 2) * w)
            y = int((cy - bh / 2) * h)
            width = int(bw * w)
            height = int(bh * h)

            boxes.append([x, y, width, height])
            confidences.append(float(confidence))
            class_ids.append(class_id)

    # NMS 적용
    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)

    for i in indices.flatten():
        x, y, w_box, h_box = boxes[i]
        conf = confidences[i]
        class_name = CLASS_NAMES[class_ids[i]]
        label = f"{class_name} {conf:.2f}"

        # 박스 + 라벨 표시
        cv2.rectangle(img, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
        cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 결과 시각화
    cv2.imshow("Detection Result", img)
    key = cv2.waitKey(0)  # 키 입력 대기
    if key == 27:  # ESC 누르면 종료
        break

cv2.destroyAllWindows()