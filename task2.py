import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# 1. 设置随机种子，保证可复现
torch.manual_seed(42)
np.random.seed(42)

# 2. 数据加载与预处理
data = pd.read_csv("task2.csv")  # 确保文件在当前目录
x_raw = data['x'].values.astype(np.float32).reshape(-1, 1)
y_raw = data['y'].values.astype(np.float32).reshape(-1, 1)

# 标准化
x_mean, x_std = x_raw.mean(), x_raw.std()
y_mean, y_std = y_raw.mean(), y_raw.std()
x_norm = (x_raw - x_mean) / x_std
y_norm = (y_raw - y_mean) / y_std

# 转换为张量并创建 DataLoader
x_tensor = torch.tensor(x_norm)
y_tensor = torch.tensor(y_norm)
dataset = TensorDataset(x_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


# 3. 构建 MLP 模型
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.layers(x)


model = MLP()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. 训练模型
epochs = [10, 100, 1000]
predictions = {}

for epoch_num in epochs:
    model = MLP()  # 每次重新初始化
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epoch_num):
        model.train()
        total_loss = 0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 100 == 0:
            print(f'Epoch {epoch + 1}/{epoch_num}, Loss: {total_loss / len(dataloader):.4f}')

    # 预测并反标准化
    model.eval()
    with torch.no_grad():
        y_pred_norm = model(x_tensor).numpy()
        y_pred = y_pred_norm * y_std + y_mean
        predictions[epoch_num] = y_pred

# 5. 可视化对比
plt.figure(figsize=(15, 5))

# 原始数据
plt.subplot(1, 3, 1)
plt.scatter(x_raw, y_raw, label='真实数据', s=10, alpha=0.6)
plt.plot(x_raw, predictions[10], 'r-', label='拟合曲线 (Epoch=10)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Epoch=10')

plt.subplot(1, 3, 2)
plt.scatter(x_raw, y_raw, label='真实数据', s=10, alpha=0.6)
plt.plot(x_raw, predictions[100], 'r-', label='拟合曲线 (Epoch=100)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Epoch=100')

plt.subplot(1, 3, 3)
plt.scatter(x_raw, y_raw, label='真实数据', s=10, alpha=0.6)
plt.plot(x_raw, predictions[1000], 'r-', label='拟合曲线 (Epoch=1000)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Epoch=1000')

plt.tight_layout()
plt.show()