import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Dataset

# 데이터 경로 설정
MNIST_ROOT = "./datasets/MNIST"

# 훈련 or 모델 테스트 설정
IS_TRAIN = False

# 이미지를 읽는 함수 정의
def get_image(p: str):
    return Image.open(p).convert("L")

# 데이터셋 클래스 정의
class MNISTDataset(Dataset):
    def __init__(self, root: str, train: bool):
        super().__init__()

        # 훈련 데이터셋 또는 검증 데이터셋의 루트 디렉토리 설정
        if train:
            self.root = os.path.join(root, "train")
        else:
            self.root = os.path.join(root, "val")

        # 데이터 리스트 생성
        data_list = []
        for i in range(10):
            dir = os.path.join(self.root, str(i))
            for img in os.listdir(dir):
                img_path = os.path.join(dir, img)
                data_list.append((i, img_path))
        self.data_list = data_list

        # 데이터 전처리
        self.transform = transforms.Compose([
            transforms.Resize((28, 28)),  # 이미지 크기를 (28, 28)로 변경
            transforms.ToTensor()
        ])

    def __len__(self):
        # 데이터셋의 총 데이터 개수 반환
        return len(self.data_list)

    def __getitem__(self, idx: int):
        # 인덱스에 해당하는 데이터 반환
        number, img_path = self.data_list[idx]

        # 이미지 파일을 PIL 객체로 읽어들이고 텐서로 변환
        img_obj = get_image(img_path)
        img_tensor = self.transform(img_obj)

        return img_tensor, number

# 모델 정의
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # 입력 데이터의 크기를 변경 (28x28 이미지를 시계열 데이터로 변환)
        x = x.view(x.size(0), x.size(2), -1)
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out

# 하이퍼파라미터 설정
input_size = 28  # 입력 데이터의 크기 (28x28)
hidden_size = 128  # GRU 레이어의 은닉 상태 크기
num_layers = 2  # GRU 레이어의 층 수
num_classes = 10  # 클래스 수 (0부터 9까지)

# 모델 생성
model = GRUModel(input_size, hidden_size, num_layers, num_classes)

if IS_TRAIN:
    # 데이터 로더 설정
    train_dataset = MNISTDataset(MNIST_ROOT, train=True)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # 손실 함수 및 옵티마이저 정의
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 학습 루프
    for epoch in range(5):
        for data, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/5], Loss: {loss.item():.4f}')

    # 모델 저장
    torch.save(model.state_dict(), 'gru_mnist_model.pth')
else:
    # 모델 불러오기
    model.load_state_dict(torch.load('gru_mnist_model.pth'))

# 테스트 데이터셋 생성
test_dataset = MNISTDataset(MNIST_ROOT, train=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 모델 테스트
model.eval()
match = 0
total = 0

with torch.no_grad():
    for data, labels in test_loader:
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        match += (predicted == labels).sum().item()

accuracy = 100 * match / total
print(f'Test Accuracy: {accuracy:.2f}%')