# resnet/preprocess.py
import os
import shutil
import random
from PIL import Image
from torchvision import transforms

raw_dir = './resnet/data/raw/'
processed_dir = './resnet/data/processed/'
classes = ['aligned', 'misaligned']

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 일반적인 이미지 크기
    transforms.ToTensor(),          # 텐서로 변환 (모델 입력용)
])

os.makedirs(processed_dir + 'train', exist_ok=True)
os.makedirs(processed_dir + 'val', exist_ok=True)

split_ratio = 0.8

for cls in classes:
    os.makedirs(processed_dir + f'train/{cls}', exist_ok=True)
    os.makedirs(processed_dir + f'val/{cls}', exist_ok=True)

    images = os.listdir(raw_dir + cls)
    random.shuffle(images)
    split_point = int(len(images) * split_ratio)

    train_images = images[:split_point]
    val_images = images[split_point:]

    for img_set, target_dir in [(train_images, 'train'), (val_images, 'val')]:
        for img_name in img_set:
            img_path = raw_dir + cls + '/' + img_name
            img = Image.open(img_path).convert('RGB')  # 이미지 색상 통일
            resized = transforms.Resize((224, 224))(img)
            resized.save(processed_dir + f'{target_dir}/{cls}/{img_name}')
