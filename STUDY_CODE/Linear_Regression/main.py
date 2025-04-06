import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
# y = 2x + 1 + noise
# 실제 x feature 과 해당하는 y 값들 랜덤하게 생성하는 부분
np.random.seed(42)
x_numpy = np.random.rand(100, 1).astype(np.float32) # 100 개 샘플 (100, 1) shape
y_numpy = 2 * x_numpy + 1 + 0.1 * np.random.randn(100, 1).astype(np.float32)

# Convert numpy arrays to PyTorch tensors
x_train = torch.from_numpy(x_numpy)
y_train = torch.from_numpy(y_numpy)

# Define a simple linear regression model
class LinearRegressionModel(nn.Module):
    def __init__(self):     # 생성자
        super(LinearRegressionModel, self).__init__()   # 부모 클래스의 생성자 호출
        self.linear = nn.Linear(1, 1)  # 1 input feature, 1 output feature -> 1개의 입력에 대해 1개의 출력을 가진다는 의미

    def forward(self, x):
        return self.linear(x)

# Initialize the model, define the loss function and the optimizer
model = LinearRegressionModel()
criterion = nn.MSELoss()  # Mean Squared Error Loss -> 손실 함수 정의
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)  # Stochastic Gradient Descent

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(x_train)    # 모델에 x_train을 넣어 예측값을 구하는 부분
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
