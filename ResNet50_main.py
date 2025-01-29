import torch
import torch.nn as nn
import torch.nn.functional as F
from CBAM import CBAM
from old_dataloader import dataloader, labels
import torchvision.models as models

class DualStreamResNet50UNet(nn.Module):
    def __init__(self, num_classes):
        super(DualStreamResNet50UNet, self).__init__()

        # 加载预训练的 ResNet50
        resnet = models.resnet50(pretrained=True)

        # RGB 流的初始层
        self.rgb_initial = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )

        # 深度图流的初始层
        self.depth_initial = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # 编码器部分（ResNet50）
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # 添加降维层，使每个流的特征在融合前具有相同的通道数
        self.reduce_rgb = nn.Conv2d(2048, 512, kernel_size=1)
        self.reduce_depth = nn.Conv2d(2048, 512, kernel_size=1)

        # 融合层
        self.fusion_conv = nn.Conv2d(1024, 512, kernel_size=1)  # 修改为 1024 输入通道
        self.fusion_cbam = CBAM(channels=512)

        # 解码器部分
        self.decoder1 = self._make_decoder(512, 256)
        self.decoder2 = self._make_decoder(256 + 256, 128)  # 考虑跳跃连接
        self.decoder3 = self._make_decoder(128 + 128, 64)   # 考虑跳跃连接
        self.decoder4 = self._make_decoder(64 + 64, 64)     # 考虑跳跃连接

        # 添加降维层用于跳跃连接
        self.reduce_enc3 = nn.Conv2d(2048, 256, kernel_size=1)
        self.reduce_enc2 = nn.Conv2d(1024, 128, kernel_size=1)
        self.reduce_enc1 = nn.Conv2d(512, 64, kernel_size=1)

        # 最终卷积层
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def _make_decoder(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, rgb, depth):
        # 处理RGB流
        rgb_features = self.rgb_initial(rgb)
        enc1_rgb = self.encoder1(rgb_features)
        enc2_rgb = self.encoder2(enc1_rgb)
        enc3_rgb = self.encoder3(enc2_rgb)
        enc4_rgb = self.encoder4(enc3_rgb)

        # 处理深度流
        depth_features = self.depth_initial(depth)
        enc1_depth = self.encoder1(depth_features)
        enc2_depth = self.encoder2(enc1_depth)
        enc3_depth = self.encoder3(enc2_depth)
        enc4_depth = self.encoder4(enc3_depth)

        # 降维
        enc4_rgb_reduced = self.reduce_rgb(enc4_rgb)
        enc4_depth_reduced = self.reduce_depth(enc4_depth)

        # 融合两个流的特征
        combined_features = torch.cat((enc4_rgb_reduced, enc4_depth_reduced), dim=1)
        combined_features = self.fusion_conv(combined_features)
        combined_features = self.fusion_cbam(combined_features)

        # 解码器
        dec1 = self.decoder1(combined_features)

        # 减少跳跃连接的通道数
        enc3_rgb_reduced = self.reduce_enc3(torch.cat([enc3_rgb, enc3_depth], dim=1))
        dec2 = self.decoder2(torch.cat([dec1, enc3_rgb_reduced], dim=1))

        enc2_rgb_reduced = self.reduce_enc2(torch.cat([enc2_rgb, enc2_depth], dim=1))
        dec3 = self.decoder3(torch.cat([dec2, enc2_rgb_reduced], dim=1))

        enc1_rgb_reduced = self.reduce_enc1(torch.cat([enc1_rgb, enc1_depth], dim=1))
        dec4 = self.decoder4(torch.cat([dec3, enc1_rgb_reduced], dim=1))

        # 最终卷积层
        output = self.final_conv(dec4)

        # 调整输出尺寸以匹配标签尺寸
        h, w = depth.size()[2:]  # 假设输入和标签尺寸一致
        output = F.interpolate(output, size=(640, 480), mode='bilinear', align_corners=True)

        return output

class mIoULoss(nn.Module):
    def __init__(self, num_classes, ignore_index=255):
        super(mIoULoss, self).__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def forward(self, inputs, target):
        # 确保输入和目标具有相同的空间尺寸
        h, w = target.size()[1:]
        inputs = F.interpolate(inputs, size=(h, w), mode='bilinear', align_corners=True)
        
        # 使用 softmax 提取概率分布
        preds = F.softmax(inputs, dim=1)

        # 创建 one-hot 标签，忽略无效类别
        valid_mask = (target != self.ignore_index).long()
        target_valid = target.clone()
        target_valid[target == self.ignore_index] = 0  # 暂时将无效类别设为0以创建one-hot编码
        
        target_onehot = F.one_hot(target_valid, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        target_onehot = target_onehot * valid_mask.unsqueeze(1)  # 排除无效类别

        # 计算交集和并集
        intersection = (preds * target_onehot).sum(dim=(2, 3))
        union = (preds + target_onehot - preds * target_onehot).sum(dim=(2, 3))

        # 计算 IoU 和 mIoU
        valid_classes = (union > 0).float()  # 只计算有有效数据的类
        iou = intersection / (union + 1e-10)
        miou = (iou * valid_classes).sum() / valid_classes.sum()

        return 1 - miou

from tqdm import tqdm

def train_model(model, dataloader, criterion, optimizer, num_epochs=25, num_classes=40):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        
        # 使用 tqdm 包装 dataloader，创建进度条
        with tqdm(dataloader, unit="batch") as tepoch:
            for images, depths, labels in tepoch:
                tepoch.set_description(f"Epoch {epoch + 1}/{num_epochs}")
                #
                labels = torch.clamp(labels, 0, num_classes - 1)

                images = images.to(device)
                depths = depths.to(device)
                labels = labels.long().to(device)

                # 零梯度
                optimizer.zero_grad()

                # 前向传播 + 后向传播 + 优化
                outputs = model(images, depths)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                
                # 更新进度条信息
                tepoch.set_postfix(loss=loss.item())
        
        avg_loss = running_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

        # 每个epoch结束后进行一次验证
        validate_model(model, dataloader, num_classes)

def validate_model(model, dataloader, num_classes):
    model.eval()
    miou_sum = 0.0
    
    # 使用 tqdm 包装 dataloader，创建进度条
    with torch.no_grad(), tqdm(dataloader, unit="batch", desc="Validation") as tepoch:
        for images, depths, labels in tepoch:
            images, depths, labels = images.to(device), depths.to(device), labels.long().to(device)
            outputs = model(images, depths)
            miou = compute_miou(outputs, labels, num_classes)
            miou_sum += miou.item()
            
            # 更新进度条信息
            tepoch.set_postfix(mIoU=miou.item())

    avg_miou = miou_sum / len(dataloader)
    print(f'Validation mIoU: {avg_miou:.4f}')

def compute_miou(preds, labels, num_classes):
    # preds: [batch_size, num_classes, H, W]
    # labels: [batch_size, H, W]

    # 调整预测结果尺寸以匹配标签尺寸
    _, h, w = labels.size()
    preds = F.interpolate(preds, size=(h, w), mode='bilinear', align_corners=True)

    # 将预测结果转换为类别索引
    preds = torch.argmax(preds, dim=1)

    # 初始化交集和并集数组
    intersection = torch.zeros(num_classes, device=preds.device)
    union = torch.zeros(num_classes, device=preds.device)

    for cls in range(num_classes):
        pred_cls = (preds == cls).float()
        label_cls = (labels == cls).float()
        intersection[cls] = (pred_cls * label_cls).sum()
        union[cls] = (pred_cls + label_cls).sum() - intersection[cls]

    # 计算 IoU 和 mIoU
    iou = intersection / (union + 1e-10)  # 添加一个小常数以防止除零
    miou = iou.mean()

    return miou

def evaluate_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, depths, labels in dataloader:
            images = images.to(device)
            depths = depths.to(device)
            labels = labels.long().to(device)

            outputs = model(images, depths)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the test set: {100 * correct / total:.2f}%')

if __name__ == "__main__":
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化模型、损失函数和优化器
    model = DualStreamResNet50UNet(num_classes=40).to(device)
    
    # 使用交叉熵损失代替 mIoU 损失
    criterion = nn.CrossEntropyLoss(ignore_index=255).to(device)  # 忽略无效类别
    optimizer = torch.optim.Adadelta(model.parameters(), lr=1e-1)

    # 开始训练
    train_model(model, dataloader, criterion, optimizer, num_epochs=10)

    # 模型评估
    evaluate_model(model, dataloader)
