import cv2
import numpy as np
import onnxruntime
import glob
import os
import random

# ----------------------------
# 설정
# ----------------------------
MODEL_PATH = "models/best.onnx"
IMAGE_DIR = "sample/"
IMG_SIZE = 640
CONF_THRESHOLD = 0.25
NMS_THRESHOLD = 0.45
CLASS_NAMES = ['door']  # 학습한 클래스 이름만 있으면 됩니다.

# ----------------------------
# Letterbox (YOLOv5 style 전처리)
# ----------------------------
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    shape = img.shape[:2]  # current shape [height, width]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return img, r, dw, dh

# ----------------------------
# 모델 로딩
# ----------------------------
session = onnxruntime.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name

# ----------------------------
# 이미지 처리
# ----------------------------

# image_paths = glob.glob(os.path.join(IMAGE_DIR, "*.jpg")) + glob.glob(os.path.join(IMAGE_DIR, "*.png"))

# 'o_'로 시작하는 파일과 'x_'로 시작하는 파일을 각각 찾습니다
o_files = glob.glob(os.path.join(IMAGE_DIR, "o_*.jpg")) + glob.glob(os.path.join(IMAGE_DIR, "o_*.png"))
x_files = glob.glob(os.path.join(IMAGE_DIR, "x_*.jpg")) + glob.glob(os.path.join(IMAGE_DIR, "x_*.png"))

# 각각 20개씩 랜덤으로 선택합니다
selected_o_files = random.sample(o_files, min(20, len(o_files)))
selected_x_files = random.sample(x_files, min(20, len(x_files)))

# 선택된 파일들을 합칩니다
image_paths = selected_o_files + selected_x_files

# 파일 개수 출력
print(f"[INFO] Selected {len(selected_o_files)} 'o_' files and {len(selected_x_files)} 'x_' files for testing")

for image_path in sorted(image_paths):
    print(f"[INFO] Processing: {image_path}")
    img_raw = cv2.imread(image_path)
    if img_raw is None:
        print(f"[WARN] Failed to read {image_path}")
        continue

    img, r, dw, dh = letterbox(img_raw, (IMG_SIZE, IMG_SIZE))

    img_input = img.astype(np.float32) / 255.0
    img_input = np.transpose(img_input, (2, 0, 1))
    img_input = np.expand_dims(img_input, axis=0)

    # ----------------------------
    # 추론
    # ----------------------------
    outputs = session.run(None, {input_name: img_input})[0]  # shape: (1, 25200, 6)
    outputs = np.squeeze(outputs)

    boxes = []
    scores = []
    class_ids = []

    for det in outputs:
        conf = det[4]
        if conf < CONF_THRESHOLD:
            continue

        class_id = int(np.argmax(det[5:]))
        score = det[5 + class_id]

        if score < CONF_THRESHOLD:
            continue

        cx, cy, w_box, h_box = det[0:4]
        x = (cx - w_box / 2 - dw) / r
        y = (cy - h_box / 2 - dh) / r
        w = w_box / r
        h = h_box / r

        boxes.append([int(x), int(y), int(w), int(h)])
        scores.append(float(score))
        class_ids.append(class_id)

    # ----------------------------
    # NMS
    # ----------------------------
    indices = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESHOLD, NMS_THRESHOLD)
    if len(indices) == 0:
        print(f"[WARN] No boxes detected for {image_path}")
        continue

    img_height, img_width = img_raw.shape[:2]
    center_x = img_width // 2

    # 세로선 좌우 기준: 중앙 -700px, 중앙 +700px
    left_line_x = center_x - 700
    right_line_x = center_x + 700

    # 세로선 그리기 (빨간색)
    cv2.line(img_raw, (left_line_x, 0), (left_line_x, img_height), (0, 0, 255), 2)
    cv2.line(img_raw, (right_line_x, 0), (right_line_x, img_height), (0, 0, 255), 2)

    for i in indices.flatten():
        x, y, w_box, h_box = boxes[i]
        label = f"{CLASS_NAMES[class_ids[i]]} {scores[i]:.2f}"
        cv2.rectangle(img_raw, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
        cv2.putText(img_raw, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # ----- 상태 판별 로직 추가 -----
        # if x <= left_line_x or x >= right_line_x:
        left_door_x = x
        right_door_x = x + w_box
        if left_door_x < left_line_x or right_door_x > right_line_x:
            status = "unposition"
            status_color = (0, 0, 255)  # 빨간색
        else:
            status = "position"
            status_color = (0, 255, 0)  # 초록색

        # 상태 출력
        print(f"[STATUS] {os.path.basename(image_path)} → {status}")

        # 이미지에 상태 표시
        cv2.putText(img_raw, f"Status: {status}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)

    # ----------------------------
    # 결과 시각화 + 중앙 기준선 추가
    # ----------------------------

    # 이미지 보여주기
    cv2.imshow("ONNX Runtime Result", img_raw)
    key = cv2.waitKey(0)
    if key == 27:  # ESC 누르면 종료
        break

cv2.destroyAllWindows()