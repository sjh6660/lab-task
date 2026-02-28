
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

try:
    from thop import profile, clever_format
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    print("警告: 未安装 thop 库，将跳过参数量和 FLOPs 计算。如需计算请运行: pip install thop")

DATA_ROOT = 'D:/datasets/cifar10'
mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ImprovedCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(ImprovedCNN, self).__init__()
        self.in_channels = 32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = self._make_layer(32, 2, stride=1)
        self.layer2 = self._make_layer(64, 2, stride=2)
        self.layer3 = self._make_layer(128, 2, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.3)

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(ResidualBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

def train_with_tricks(use_mixup=True, use_warmup=True):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pin_memory = True if torch.cuda.is_available() else False
    print(f"使用设备: {device}")

    trainset = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=True, download=False, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True,
                                              num_workers=2, pin_memory=pin_memory)

    testset = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=False, download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False,
                                             num_workers=2, pin_memory=pin_memory)

    net = ImprovedCNN().to(device)

    if THOP_AVAILABLE:
        dummy_input = torch.randn(1, 3, 32, 32).to(device)
        flops, params = profile(net, inputs=(dummy_input,), verbose=False)
        flops, params = clever_format([flops, params], "%.3f")
        print(f"模型参数量: {params}, FLOPs: {flops}")
    else:
        print("未计算参数量和 FLOPs（thop 未安装）")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=5e-4)

    if use_warmup:
        def warmup_lambda(epoch):
            if epoch < 5:
                return (epoch + 1) / 5
            else:
                return 0.5 * (1 + np.cos(np.pi * (epoch - 5) / (30 - 5)))
        scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
    else:
        scheduler = None

    train_losses, test_accuracies = [], []
    print("开始训练...")
    for epoch in range(30):
        net.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            if use_mixup:
                inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, alpha=1.0)
                outputs = net(inputs)
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            else:
                outputs = net(inputs)
                loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        if scheduler:
            scheduler.step()
        epoch_loss = running_loss / len(trainloader)
        train_losses.append(epoch_loss)

        correct, total = 0, 0
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

        lr = scheduler.get_last_lr()[0] if scheduler else 0.001
        print(f'Epoch {epoch+1:2d} | Loss: {epoch_loss:.4f} | Test Acc: {accuracy:.2f}% | LR: {lr:.5f}')

    print("训练完成！")
    return net, train_losses, test_accuracies

if __name__ == '__main__':
    net, losses, accs = train_with_tricks(use_mixup=True, use_warmup=True)

    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(range(1,31), losses, 'b-', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve (Optimized)')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(range(1,31), accs, 'orange', label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Test Accuracy Curve (Optimized)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('optimized_curves.png')
    plt.show()