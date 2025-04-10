import os
import cv2
from pathlib import Path

# 설정
VIDEO_PATH = "output.mp4"
OUTPUT_BASE = Path("output_all_frames")
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

INTERVAL_SEC = 0.2  # 0.5초 간격
FPS = None

# 분류 디렉토리
categories = {
    'o': "o",
    'x': "x",
}

for folder in categories.values():
    (OUTPUT_BASE / folder).mkdir(parents=True, exist_ok=True)

# 비디오 로드
cap = cv2.VideoCapture(VIDEO_PATH)
FPS = cap.get(cv2.CAP_PROP_FPS)
frame_interval = int(FPS * INTERVAL_SEC)
frame_id = 0

print("👀 키보드로 분류하세요: [o] 정위치 / [x] 비정위치 / [s] 건너뛰기 / [q] 종료")
# print("👀 키보드로 분류하세요: [o] 열림 / [c] 닫힘 / [s] 건너뛰기 / [q] 종료")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    if current_frame % frame_interval != 0:
        continue

    # 프레임을 시계 방향으로 90도 회전
    # rotated_frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    rotated_frame = frame

    preview = rotated_frame.copy()
    cv2.putText(preview, f"Frame {frame_id}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Label Frame", preview)

    key = cv2.waitKey(0) & 0xFF
    # if key == ord('q'):
    #     print("🚪 종료")
    #     break
    # elif key in map(ord, categories.keys()):
    #     label = categories[chr(key)]
    #     out_path = OUTPUT_BASE / label / f"{label}_{frame_id:04d}.jpg"
    #     cv2.imwrite(str(out_path), rotated_frame)
    #     print(f"✅ 저장됨: {out_path}")
    #     frame_id += 1
    # elif key == ord('s'):
    #     print("⏭️ 건너뜀")
    #     frame_id += 1
    #     continue
    label = "o"
    out_path = OUTPUT_BASE / label / f"{label}_{frame_id:04d}.jpg"
    cv2.imwrite(str(out_path), rotated_frame)
    print(f"✅ 저장됨: {out_path}")

    frame_id += 1

cap.release()
cv2.destroyAllWindows()