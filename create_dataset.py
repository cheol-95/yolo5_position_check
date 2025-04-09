import os
import shutil
import random

# 경로 설정
base_dir = 'cropped_images'
train_dir = 'train'
val_dir = 'val'
test_dir = 'test'

# 각 폴더 생성
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# 데이터셋 비율 설정
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# 각 클래스에 대해 이미지 나누기
for class_name in ['o', 'x']:
    class_dir = os.path.join(base_dir, class_name)
    images = os.listdir(class_dir)
    random.shuffle(images)

    train_count = int(len(images) * train_ratio)
    val_count = int(len(images) * val_ratio)

    for i, image in enumerate(images):
        src_path = os.path.join(class_dir, image)
        if i < train_count:
            dst_dir = os.path.join(train_dir, class_name)
        elif i < train_count + val_count:
            dst_dir = os.path.join(val_dir, class_name)
        else:
            dst_dir = os.path.join(test_dir, class_name)

        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy(src_path, os.path.join(dst_dir, image))
