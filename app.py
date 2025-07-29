import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import gradio as gr

# 1. 모델 클래스 정의 (학습 때와 동일해야 함)
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
	
# 2. 클래스 이름 정의
classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 3. 모델 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FashionCNN().to(device)
model.load_state_dict(torch.load("fashion_cnn.pth", map_location=device))
model.eval()

# 4. 예측 함수
def predict(img: Image.Image):
	transform = transforms.Compose([
		transforms.Grayscale(num_output_channels=1),
		transforms.Resize((28, 28)),
		transforms.ToTensor(),
		transforms.Normalize((0.5,), (0.5,))
	])
	img_tensor = transform(img).unsqueeze(0).to(device)

	with torch.no_grad():
		output = model(img_tensor)
		pred = output.argmax(dim=1).item()

	return f"예측 결과 : {classes[pred]}"

demo = gr.Interface(
	fn=predict,
	inputs=gr.Image(type="pil"),
	outputs="text",
	title="Fashion MNIST 의류 분류기",
	description="28x28 흑백 이미지로 테스트해보세요!"
)

# 6. 실행
demo.launch()