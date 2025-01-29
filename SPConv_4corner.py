import torch
import torch.nn as nn
import torch.nn.functional as F

class SpecialConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, spacing=18, sigma=2.0, truncate=9.0):
        super(SpecialConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.spacing = spacing  # 稀疏采样点之间的间距
        self.sigma = sigma
        self.truncate = truncate

        # 定义常规卷积层
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            bias=False
        )

        # 初始化核函数权重
        self.rbf_weights = self._generate_rbf_weights()

    def _generate_rbf_weights(self):
        """
        生成核函数权重矩阵，包括中心、四个角和四个边缘中点
        """
        kernel_size = self.kernel_size
        center = (kernel_size - 1) // 2  # 卷积核中心

        # 使用 meshgrid 生成坐标
        x = torch.arange(kernel_size).float() - center
        y = torch.arange(kernel_size).float() - center
        xx, yy = torch.meshgrid(x, y, indexing='ij')

        # 定义所有中心点：中心、四个角和四个边缘中点
        centers = [
            (center, center),  # 中心
            (0, 0),  # 左上角
            (0, kernel_size - 1),  # 右上角
            (kernel_size - 1, 0),  # 左下角
            (kernel_size - 1, kernel_size - 1),  # 右下角
            (center, 0),  # 左边缘中点
            (center, kernel_size - 1),  # 右边缘中点
            (0, center),  # 上边缘中点
            (kernel_size - 1, center)  # 下边缘中点
        ]

        # 初始化权重矩阵
        weights = torch.zeros_like(xx)

        # 为每个中心点生成核函数权重
        for cx, cy in centers:
            distance = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            rbf = torch.exp(-(distance ** 2) / (2 * self.sigma ** 2))
            rbf[distance > self.truncate] = 0.0  # 超过截断阈值的权重为0
            weights += rbf  # 将每个核函数的权重叠加

        # 根据 spacing 设置稀疏采样点的权重
        for i in range(0, kernel_size, self.spacing):
            for j in range(0, kernel_size, self.spacing):
                weights[i, j] = 1.0  # 强制采样点权重为1

        weights = weights / weights.sum()  # 归一化权重

        return weights.view(1, 1, kernel_size, kernel_size)  # 调整形状以适配卷积

    def forward(self, x):
        """
        前向传播
        """
        # 原始卷积权重
        original_weights = self.conv.weight

        # 核函数权重应用到特征点上
        rbf_weights = self.rbf_weights.to(x.device)
        weighted_conv_weight = original_weights * rbf_weights

        # 应用加权后的卷积
        output = F.conv2d(x, weighted_conv_weight, stride=self.stride, padding=(self.kernel_size - 1) // 2)
        return output

# 测试代码
if __name__ == "__main__":
    # 输入数据 (batch_size, channels, height, width)
    input_tensor = torch.randn(1, 64, 32, 32)

    # 创建卷积模块
    conv = SpecialConv2d(
        in_channels=64,
        out_channels=64,
        kernel_size=3,
        stride=1,
        spacing=18,  # 稀疏采样点之间的间距
        sigma=1.0,
        truncate=2.0
    )

    # 前向传播
    output_tensor = conv(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)