import h5py
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# 加载 NYUv2 数据
file_path = 'F:/ICML2025/asym/demoNet/src/nyu_depth_v2_labeled.mat'
with h5py.File(file_path, 'r') as f:
    images = np.array(f['images'])  # (N, C, H, W)
    depths = np.array(f['depths'])  # (N, H, W)
    labels = np.array(f['labels'])  # (N, H, W)

# 打印原始数据形状
print(f"Original Images shape: {images.shape}")
print(f"Original Depths shape: {depths.shape}")
print(f"Original Labels shape: {labels.shape}")


# 数据预处理函数
def preprocess_data(image, depth, label):
    transform_image = transforms.Compose([
        transforms.ToTensor(),  # 转换为 (C, H, W)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 确保输入到 ToTensor 的图像是 (H, W, C)
    if image.shape[0] == 3:  # 输入为 (C, H, W)
        image = image.transpose(1, 2, 0)  # 转换为 (H, W, C)
    
    image = transform_image(image)  # 转换为 (C, H, W)
    depth = torch.tensor(depth, dtype=torch.float32).unsqueeze(0) / np.max(depth)  # 转换为 (1, H, W)
    label = torch.tensor(label, dtype=torch.int64)  # 保持 (H, W)
    return image, depth, label

class NYUv2Dataset(Dataset):
    def __init__(self, images, depths, labels):
        self.images = images  # (N, C, H, W)
        self.depths = depths  # (N, H, W)
        self.labels = labels  # (N, H, W)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 提取单个样本
        image = self.images[idx]  # (C, H, W)
        depth = self.depths[idx]  # (H, W)
        label = self.labels[idx]  # (H, W)

        # 数据预处理
        image, depth, label = preprocess_data(image, depth, label)
        
        # 调试输出形状
        if idx == 0:  # 仅打印一次
            print(f"Image shape after preprocessing: {image.shape}")  # (3, H, W)
            print(f"Depth shape after preprocessing: {depth.shape}")  # (1, H, W)
            print(f"Label shape after preprocessing: {label.shape}")  # (H, W)
        return image, depth, label

# 创建 Dataset 和 DataLoader
dataset = NYUv2Dataset(images, depths, labels)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# 检查批量数据形状
for batch in dataloader:
    images, depths, labels = batch
    print(f"Batch images shape: {images.shape}")  # (16, 3, H, W)
    print(f"Batch depths shape: {depths.shape}")  # (16, 1, H, W)
    print(f"Batch labels shape: {labels.shape}")  # (16, H, W)
    break

