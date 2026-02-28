"""
CIFAR-10 图像分类（修正版）
数据集路径: D:/datasets/cifar10 (手动下载，无需联网)
模型: 简单 CNN，测试准确率约 75%~78%
注意：Windows 下必须将主程序放在 if __name__ == '__main__': 内，否则多进程会报错。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# 1. 数据预处理与加载（使用本地数据集）
DATA_ROOT = 'D:/datasets/cifar10'   # 数据集根目录（包含 cifar-10-batches-py 子文件夹）

# CIFAR-10 的均值与标准差（用于归一化）
mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),          # 数据增强：随机水平翻转
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# 2. 模型定义（简单 CNN）
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 32x32 -> 16x16
        x = self.pool(F.relu(self.conv2(x)))  # 16x16 -> 8x8
        x = self.pool(F.relu(self.conv3(x)))  # 8x8 -> 4x4
        x = x.view(-1, 128 * 4 * 4)            # 展平
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 3. 主程序入口（必须放在 if __name__ == '__main__' 内）
if __name__ == '__main__':
    # 加载数据集
    trainset = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=True, download=False, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=False, download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # 检查 GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    net = SimpleCNN().to(device)

    # 损失函数与优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    # 训练记录
    train_losses = []
    test_accuracies = []

    print("开始训练...")
    for epoch in range(30):  # 训练 30 个 epoch
        running_loss = 0.0
        net.train()
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(trainloader)
        train_losses.append(epoch_loss)

        # 测试集准确率
        correct = 0
        total = 0
        net.eval()
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100.0 * correct / total
        test_accuracies.append(accuracy)

        print(f'Epoch {epoch+1:2d} | Loss: {epoch_loss:.4f} | Test Acc: {accuracy:.2f}%')

    print("训练完成！")

    # 绘制 Loss 曲线与测试准确率曲线
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(range(1, 31), train_losses, 'b-', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(range(1, 31), test_accuracies, 'orange', label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Test Accuracy Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig('training_curves.png')  # 保存图片
    plt.show()

    # 可视化正确与错误分类的图片示例
    def imshow(img):
        """反标准化并显示图像"""
        img = img * torch.tensor(std).view(3,1,1) + torch.tensor(mean).view(3,1,1)
        img = img.clamp(0, 1)
        return img

    # 从测试集随机取一批数据
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    images, labels = images.to(device), labels.to(device)

    # 模型预测
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)

    # 移回 CPU 并转换为 numpy
    images_cpu = images.cpu()
    labels_cpu = labels.cpu()
    predicted_cpu = predicted.cpu()

    # 找出正确和错误分类的索引（处理可能出现的标量情况）
    correct_indices = (predicted_cpu == labels_cpu).nonzero(as_tuple=True)[0].tolist()
    incorrect_indices = (predicted_cpu != labels_cpu).nonzero(as_tuple=True)[0].tolist()

    # 显示正确分类的图片（最多5张）
    plt.figure(figsize=(15,6))
    for i, idx in enumerate(correct_indices[:5]):
        plt.subplot(2,5,i+1)
        img = imshow(images_cpu[idx])
        plt.imshow(np.transpose(img.numpy(), (1,2,0)))
        plt.title(f'True: {classes[labels_cpu[idx]]}\nPred: {classes[predicted_cpu[idx]]}')
        plt.axis('off')

    # 显示错误分类的图片（最多5张）
    for i, idx in enumerate(incorrect_indices[:5]):
        plt.subplot(2,5,i+6)
        img = imshow(images_cpu[idx])
        plt.imshow(np.transpose(img.numpy(), (1,2,0)))
        plt.title(f'True: {classes[labels_cpu[idx]]}\nPred: {classes[predicted_cpu[idx]]}', color='red')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('predictions.png')
    plt.show()

    # 最终测试准确率（再测一次完整测试集）
    correct_total = 0
    total_total = 0
    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            total_total += labels.size(0)
            correct_total += (predicted == labels).sum().item()
    final_acc = 100.0 * correct_total / total_total
    print(f"最终测试集准确率: {final_acc:.2f}%")