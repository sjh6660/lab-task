import os
import sys
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import v2 as T
from thop import profile, clever_format

print("===== 脚本开始执行 =====")
print(f"当前工作目录: {os.getcwd()}")

# ==================== 配置参数 ====================
# 数据集路径（使用相对路径，假设 PennFudanPed 位于脚本所在目录下）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "PennFudanPed")   # 数据集主文件夹
print(f"数据集路径: {DATA_DIR}")

NUM_EPOCHS = 10
BATCH_SIZE = 2
LEARNING_RATE = 0.005
USE_WARMUP = True          # 是否使用学习率预热
USE_MIXUP = False          # 实例分割中MixUp较复杂，暂不实现
# =================================================

# -------------------- 检查本地数据集 --------------------
def check_dataset(data_dir):
    """检查数据集是否存在且包含必要的子文件夹"""
    if not os.path.exists(data_dir):
        print(f"错误：数据集文件夹不存在于 {data_dir}")
        print("请确保 PennFudanPed 文件夹放在正确位置。")
        return False
    png_dir = os.path.join(data_dir, "PNGImages")
    mask_dir = os.path.join(data_dir, "PedMasks")
    if not os.path.exists(png_dir):
        print(f"错误：找不到 PNGImages 文件夹（应位于 {png_dir}）")
        return False
    if not os.path.exists(mask_dir):
        print(f"错误：找不到 PedMasks 文件夹（应位于 {mask_dir}）")
        return False
    print("数据集检查通过。")
    return True

if not check_dataset(DATA_DIR):
    sys.exit(1)

# -------------------- 数据集定义 --------------------
class PennFudanDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))
        print(f"数据集加载: {len(self.imgs)} 张图片")

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        mask = np.array(mask)
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]  # 去除背景 (0)

        masks = mask == obj_ids[:, None, None]  # [num_objs, H, W]

        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)  # 只有一类：行人
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

# -------------------- 数据变换 --------------------
def get_transform(train):
    transforms = []
    transforms.append(T.ToImage())
    transforms.append(T.ToDtype(torch.float32, scale=True))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)

# -------------------- 模型获取 --------------------
def get_model_instance_segmentation(num_classes):
    print("正在加载预训练 Mask R-CNN 模型...")
    model = maskrcnn_resnet50_fpn(weights='DEFAULT')
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    print("模型加载完成")
    return model

# -------------------- MixUp 占位 --------------------
def mixup(images, targets, alpha=1.0):
    return images, targets

# -------------------- 训练一个epoch --------------------
def train_one_epoch(model, optimizer, data_loader, device, epoch, warmup_scheduler=None):
    model.train()
    total_loss = 0
    for i, (images, targets) in enumerate(data_loader):
        if USE_MIXUP:
            images, targets = mixup(images, targets)

        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        total_loss += losses.item()

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if warmup_scheduler is not None:
            warmup_scheduler.step()

        if i % 10 == 0:
            print(f"  step {i}, loss: {losses.item():.4f}")

    return total_loss / len(data_loader)

# -------------------- 主函数 --------------------
def main():
    print("\n===== 主函数开始执行 =====")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"使用设备: {device}")

    num_classes = 2  # 背景 + 行人

    # 加载数据集
    print("加载训练集...")
    dataset = PennFudanDataset(DATA_DIR, transforms=get_transform(train=True))
    print("加载测试集...")
    dataset_test = PennFudanDataset(DATA_DIR, transforms=get_transform(train=False))

    # 划分训练/测试集（随机取30张作为测试）
    indices = torch.randperm(len(dataset)).tolist()
    train_dataset = torch.utils.data.Subset(dataset, indices[:-30])
    test_dataset = torch.utils.data.Subset(dataset_test, indices[-30:])
    print(f"训练集大小: {len(train_dataset)}, 测试集大小: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=lambda x: tuple(zip(*x))
    )
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        collate_fn=lambda x: tuple(zip(*x))
    )

    # 初始化模型
    model = get_model_instance_segmentation(num_classes)
    model.to(device)

    # 计算参数量和FLOPs
    print("计算模型参数量和FLOPs...")
    dummy_input = torch.randn(1, 3, 800, 800).to(device)
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")
    print(f"模型参数量: {params}, 计算量: {flops}")

    # 优化器
    params_to_train = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params_to_train, lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)

    # 学习率调度器
    main_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Warmup 设置
    warmup_scheduler = None
    if USE_WARMUP:
        warmup_iters = min(100, len(train_loader))
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_iters
        )
        print(f"使用Warmup，预热{warmup_iters}个step")

    # 训练循环
    print("开始训练...")
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"Epoch {epoch}/{NUM_EPOCHS}")
        loss_avg = train_one_epoch(model, optimizer, train_loader, device, epoch, warmup_scheduler)
        main_lr_scheduler.step()
        print(f"Epoch {epoch:02d} 完成 | 平均损失: {loss_avg:.4f} | 学习率: {optimizer.param_groups[0]['lr']:.6f}")

    # 保存模型
    torch.save(model.state_dict(), 'maskrcnn_pennfudan.pth')
    print("模型已保存为 maskrcnn_pennfudan.pth")

    # -------------------- 可视化 --------------------
    def visualize_prediction(model, dataset, idx, device, threshold=0.7):
        model.eval()
        img, _ = dataset[idx]
        img_tensor = img.unsqueeze(0).to(device)

        with torch.no_grad():
            prediction = model(img_tensor)[0]

        img_np = img.mul(255).permute(1, 2, 0).byte().cpu().numpy()

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(img_np)
        plt.title("原始图像")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(img_np)

        masks = prediction['masks'].cpu().numpy()
        boxes = prediction['boxes'].cpu().numpy()
        scores = prediction['scores'].cpu().numpy()

        for mask, box, score in zip(masks, boxes, scores):
            if score > threshold:
                mask_np = mask[0] > 0.5
                plt.imshow(mask_np, alpha=0.3, cmap='Reds', vmin=0, vmax=1)

                x1, y1, x2, y2 = box
                rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color='red', linewidth=2)
                plt.gca().add_patch(rect)
                plt.text(x1, y1, f"{score:.2f}", color='white', fontsize=8,
                         bbox=dict(facecolor='red', alpha=0.5))

        plt.title("预测结果 (边界框+掩码)")
        plt.axis('off')
        plt.show()

    print("显示测试集第一张图片的预测结果：")
    visualize_prediction(model, test_dataset, idx=0, device=device)

    # -------------------- 思考部分 --------------------
    print("\n" + "="*60)
    print("思考：分割任务的数据处理与分类任务有何不同？")
    print("-"*60)
    print("1. 标签形式不同：分类任务标签是类别索引（标量），而分割任务标签是与原图同尺寸的掩码矩阵（每个像素标注类别或实例ID）。")
    print("2. 实例分割需要同时检测并分割每个对象，因此数据加载时需要将彩色掩码图解析为多个二值掩码（每个实例一个），并同步计算边界框。")
    print("3. 难点：")
    print("   - 掩码文件中不同实例用不同像素值编码，需用 np.unique 提取实例ID。")
    print("   - 必须保证图像与掩码文件名一一对应（通过排序对齐）。")
    print("   - 每张图片的实例数量不同，DataLoader 需自定义 collate_fn 处理变长数据。")
    print("4. 解决方法：")
    print("   - 在 Dataset 的 __getitem__ 中动态解析掩码，生成 masks 张量和 boxes。")
    print("   - 使用 torchvision 提供的目标字典格式。")
    print("   - collate_fn 使用 lambda x: tuple(zip(*x)) 将样本组成列表。")
    print("="*60)

    print("\n===== 主函数执行完毕 =====")

if __name__ == "__main__":
    print("检测到 __name__ == '__main__'，即将调用 main()")
    main()
else:
    print("警告：__name__ 不是 '__main__'，当前值为:", __name__)