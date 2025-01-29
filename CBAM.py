import torch
import torch.nn as nn
import torch.nn.functional as F
#from torchvision.models import ghostnet

#此处是简化版CBAM模块
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM,self).__init__()
        #通道注意力机制
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, 1, bias = False)
        self.relu = nn.ReLU(inplace = True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, 1, bias = False)
        self.sigmoid = nn.Sigmoid()

        #空间注意力部分
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias= False)

    def forward(self, x):
        #通道
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        channel_attention = self.sigmoid(avg_out + max_out)
        x = x * channel_attention

        #空间
        avg_out = torch.mean(x, dim=1, keepdim = True)
        max_out, _ = torch.max(x, dim=1, keepdim = True)
        spatial_attention = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        x = x * spatial_attention

        return x


