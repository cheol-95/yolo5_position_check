# import cv2
# import numpy as np

# def check_doors_by_frame(image_path, debug=True, tolerance=15):
#     img = cv2.imread(image_path)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # 이진화
#     _, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)

#     # 윤곽선 검출
#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     h, w = thresh.shape
#     image_center_x = w // 2
#     door_edges_x = []

#     for cnt in contours:
#         x, y, box_w, box_h = cv2.boundingRect(cnt)
#         area = cv2.contourArea(cnt)
#         aspect_ratio = box_h / float(box_w)

#         # ✅ 개선된 문틀 조건
#         is_tall = aspect_ratio > 2.5
#         is_not_too_thin = box_w > 10
#         is_in_upper_half = y + box_h < h * 0.85
#         is_large_enough = area > 800

#         if is_tall and is_not_too_thin and is_in_upper_half and is_large_enough:
#             door_edges_x.append(x)             # 왼쪽 문틀
#             door_edges_x.append(x + box_w)     # 오른쪽 문틀

#             if debug:
#                 cv2.rectangle(img, (x, y), (x + box_w, y + box_h), (0, 255, 0), 2)
#                 cv2.putText(img, f"{int(aspect_ratio)}", (x, y - 5),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

#     if len(door_edges_x) < 2:
#         return "문틀 인식 실패"

#     # 문틀 중심 계산
#     door_edges_x.sort()
#     left_edge = door_edges_x[0]
#     right_edge = door_edges_x[-1]
#     doors_center = (left_edge + right_edge) // 2
#     deviation = abs(doors_center - image_center_x)

#     if debug:
#         cv2.line(img, (image_center_x, 0), (image_center_x, h), (0, 0, 255), 2)       # 이미지 중앙
#         cv2.line(img, (left_edge, 0), (left_edge, h), (255, 0, 0), 1)                # 왼쪽 문틀
#         cv2.line(img, (right_edge, 0), (right_edge, h), (255, 0, 0), 1)              # 오른쪽 문틀
#         cv2.line(img, (doors_center, 0), (doors_center, h), (0, 255, 255), 2)        # 계산된 중심

#         cv2.putText(img, f"Center Deviation: {deviation}px", (10, h - 20),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6,
#                     (0, 255, 0) if deviation < tolerance else (0, 0, 255), 2)

#         cv2.imshow("Door Frame Debug", img)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()

#     return "정위치" if deviation < tolerance else "정위치 아님"

# # 예시 사용
# print(check_doors_by_frame("./cropped_images/o/o_0032.jpg"))  # 이미지1
# print(check_doors_by_frame("./cropped_images/x/x_0058.jpg"))  # 이미지2

import cv2
import numpy as np

# === 설정 ===
image_path = './cropped_images/x/x_0058.jpg'
tolerance = 10  # 중앙으로부터 허용 오차 (픽셀)
debug = True  # 디버그 모드

# === 이미지 불러오기 ===
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# === 에지 검출 ===
edges = cv2.Canny(gray, 50, 150)

# === 윤곽선 검출 ===
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

door_frames = []

# === 문틀 후보 찾기 ===
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = h / float(w)
    
    if 1.5 < aspect_ratio < 6 and 50 < h < 300 and 10 < w < 80:
        door_frames.append((x, y, w, h))

# === 좌우 문틀 정렬 (x 기준) ===
door_frames = sorted(door_frames, key=lambda box: box[0])

# === 기준 중앙 계산 ===
image_center_x = image.shape[1] // 2

# === 문 중심 계산 ===
is_aligned = False
status_text = "문틀 인식 실패"

if len(door_frames) >= 2:
    left_frame = door_frames[0]
    right_frame = door_frames[1]

    left_center = left_frame[0] + left_frame[2] // 2
    right_center = right_frame[0] + right_frame[2] // 2
    door_center = (left_center + right_center) // 2

    if abs(door_center - image_center_x) <= tolerance:
        is_aligned = True
        status_text = "정위치"
    else:
        status_text = "위치 안맞음"

    # === 디버그 시각화 ===
    if debug:
        # 문틀 사각형
        cv2.rectangle(image, (left_frame[0], left_frame[1]), (left_frame[0]+left_frame[2], left_frame[1]+left_frame[3]), (0,255,0), 2)
        cv2.rectangle(image, (right_frame[0], right_frame[1]), (right_frame[0]+right_frame[2], right_frame[1]+right_frame[3]), (0,255,0), 2)
        
        # 문틀 중심선
        cv2.line(image, (door_center, 0), (door_center, image.shape[0]), (255, 255, 0), 1)
        
        # 이미지 중앙선
        cv2.line(image, (image_center_x, 0), (image_center_x, image.shape[0]), (0, 0, 255), 1)

        # 중심점
        cv2.circle(image, (left_center, left_frame[1]+left_frame[3]//2), 5, (255, 0, 0), -1)
        cv2.circle(image, (right_center, right_frame[1]+right_frame[3]//2), 5, (255, 0, 0), -1)

# === 결과 출력 ===
print("열차 문 상태:", status_text)

# === 디버그 이미지 보기 ===
if debug:
    cv2.putText(image, status_text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.imshow("Door Alignment Debug", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()