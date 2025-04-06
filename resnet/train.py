# train.py

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt
from IPython.display import clear_output

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 전처리
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 검증은 증강 없이 단순히 정규화
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 적용
train_dataset = datasets.ImageFolder(root='./resnet/data/processed/train', transform=train_transform)
val_dataset = datasets.ImageFolder(root='./resnet/data/processed/val', transform=val_transform)


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 최신 방식으로 사전학습된 모델 불러오기
model = resnet18(weights=ResNet18_Weights.DEFAULT)


# 기존 레이어들 고정 (원하면 전체 학습할 수도 있음)
for param in model.parameters():
    param.requires_grad = True

# 마지막 레이어만 수정 (2개 클래스: aligned, misaligned)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

model = model.to(device)

# 손실 함수 & 옵티마이저
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.0001)  # 마지막 레이어만 학습할 거라 학습률 낮게

# 인터랙티브 모드 활성화
plt.ion()

# 그래프 초기화
plt.figure(figsize=(10, 5))

# 학습 루프
epochs = 30
train_losses = []
val_losses = []

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_train_loss:.4f}')
    
    # validation 평가
    model.eval()
    correct = 0
    total = 0
    val_loss = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    print(f'Validation Loss: {avg_val_loss:.4f}')
    print(f'Validation Accuracy: {100 * correct / total:.2f}%')

    # 그래프 업데이트
    plt.clf()
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.draw()
    plt.pause(0.001)

# 모델 저장
os.makedirs('./resnet/models', exist_ok=True)
torch.save(model.state_dict(), './resnet/models/door_position_resnet18.pth')

# 인터랙티브 모드 비활성화
plt.ioff()
plt.show()
