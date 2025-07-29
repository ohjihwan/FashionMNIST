import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# 1. 데이터셋 클래스 정의
class FashionDataset(Dataset):
	def __init__(self, df):
		self.labels = df.iloc[:, 0].values
		self.images = df.iloc[:, 1:].values.reshape(-1, 1, 28, 28).astype(np.float32) / 255.0

	def __len__(self):
		return len(self.labels)
	
	def __getitem__(self, idx):
		return torch.tensor(self.images[idx]), torch.tensor(self.labels[idx])
	
# 2. 모델 정의
class FashionCNN(nn.Module):
	def __init__(self):
		super(FashionCNN, self).__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(1, 32, kernel_size=3, padding='same'),
			nn.ReLU(),
			nn.MaxPool2d(2),
			nn.Conv2d(32, 64, kernel_size=3, padding='same'),
			nn.ReLU(),
			nn.MaxPool2d(2)
		)
		self.fc = nn.Sequential(
			nn.Flatten(),
			nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
			nn.Linear(128, 10)
		)
	def forward(self, X):
		X = self.conv(X)
		X = self.fc(X)
		return X
	
# 3. CSV 로드
train_df = pd.read_csv("archive/fashion-mnist_train.csv")
test_df = pd.read_csv("archive/fashion-mnist_test.csv")

train_dataset = FashionDataset(train_df)
test_dataset = FashionDataset(test_df)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

# 4. 모델 학습
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FashionCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


print("=================")
print("모델 학습 시작...!")
print("=================")
num_epochs = 10
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

	print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}")

# 5. 저장
torch.save(model.state_dict(), "fashion_cnn.pth")
print("모델 저장 완료 : fashion_cnn.pth")