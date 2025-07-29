import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 1. 전처리 정의 (흑백 정규화)
transform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((0.5,), (0.5,))
])

# 2. 데이터셋 로드 (ubyte 포맷 자동 처리)
train_dataset = datasets.FashionMNIST(
	root='./data',
	train=True,
	download=True,
	transform=transform
)

test_dataset = datasets.FashionMNIST(
	root='./data',
	train=False,
	download=True,
	transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

# 3. 모델 정의
class FashionCNN(nn.Module):
	def __init__(self):
		super(FashionCNN, self).__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(1, 32, kernel_size=3, padding='same'),
			nn.ReLU(),
			nn.MaxPool2d(2),
			nn.Conv2d(32, 64, kernel_size=3, padding="same"),
			nn.ReLU(),
			nn.MaxPool2d(2)
		)
		self.fc = nn.Sequential(
			nn.Flatten(),
			nn.Linear(64 * 7 * 7, 128),
			nn.ReLU(),
			nn.Linear(128, 10)
		)

	def forward(self, X):
		X = self.conv(X)
		X = self.fc(X)
		return X
	
# 4. 학습 준비
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FashionCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. 학습 루프
print("=================")
print("모델 학습 시작...!")
print("=================")
num_epochs = 30
for epoch in range(num_epochs):
	model.train()
	running_loss = 0.0
	for images, labels in train_loader:
		images, labels = images.to(device), labels.to(device)

		optimizer.zero_grad()
		outputs = model(images)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()

		running_loss += loss.item()

	print(f'Epoch {epoch+1}/{num_epochs}, Loss {running_loss / len(train_loader):.4f}')

torch.save(model.state_dict(), "fashion_cnn.pth")
print("모델 저장 완료 fashion_cnn.pth")