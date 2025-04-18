import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

if torch.mps.is_available():    # 애플 실리콘 GPU 사용 가능하면
    device = torch.device("mps")
    print("Using Apple Silicon GPU (MPS) for computation.")
elif torch.cuda.is_available(): # NVIDIA GPU 사용 가능하면
    device = torch.device("cuda")
    print("Using NVIDIA GPU for computation.")
else:   # CPU 사용
    device = torch.device("cpu")
    print("Using CPU for computation.")

# y = 2x + 1 + noise
# 실제 x feature 과 해당하는 y 값들 노이즈 추가해서 랜덤하게 생성
np.random.seed(42)
x_numpy = np.random.rand(100, 1).astype(np.float32) # 100 개 샘플 (100, 1) shape
y_numpy = 2 * x_numpy + 1 + 0.1 * np.random.randn(100, 1).astype(np.float32)

# Convert numpy arrays to PyTorch tensors
x_train = torch.from_numpy(x_numpy)
y_train = torch.from_numpy(y_numpy)

# 선형 회귀 모델 정의 클래스
class LinearRegressionModel(nn.Module):     # nn.Module 상속받음
    # nn.Module 상속받은 클래스는 반드시 __init__() 메서드와 forward() 메서드를 구현해야 함

    # __init__() 메서드는 모델의 레이어를 정의하는 부분
    def __init__(self):     # 파이썬에서 __init__() 메서드는 생성자
        super().__init__()  # 부모 클래스의 생성자 호출
        self.linear = nn.Linear(1, 1)  # 1차원 입력을 받아 1차원 출력을 내는 선형 레이어 정의

    # forward() 메서드는 모델의 순전파 계산을 정의하는 부분

    def forward(self, x):
        return self.linear(x)   # 선형 레이어에 x를 넣어 예측값을 구하는 부분

# Initialize the model, define the loss function and the optimizer
model = LinearRegressionModel()
criterion = nn.MSELoss()  # Mean Squared Error Loss -> 손실 함수 정의
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)  # 학습률 0.1로 SGD 옵티마이저 정의

# 학습
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(x_train)    # 모델에 x_train을 넣어 예측값을 구하는 부분 -> forward() 메서드 호출
    loss = criterion(outputs, y_train)  # 손실 계산

    # Backward pass and optimization
    optimizer.zero_grad()   # 기울기 초기화
    loss.backward() # 기울기 계산
    optimizer.step() # 가중치 업데이트

    # Print loss every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Extract learned parameters
[w, b] = model.parameters()
print(f'Learned weight: {w.item():.4f}, Learned bias: {b.item():.4f}')

# Plot the results
'''
predicted = model(x_train).detach().numpy()  # Detach from the computation graph for plotting
plt.scatter(x_numpy, y_numpy, label='Original Data', color='blue')
plt.plot(x_numpy, predicted, label='Fitted Line', color='red')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression with PyTorch')
plt.show()
'''
