import torch
from torch import nn
import torch.nn.functional as F

class InceptionA(nn.Module):
    def __init__(self,in_channels):
        super(InceptionA, self).__init__()
        # 二维的卷积层 输入 inchannel 通道 输出16个通道 卷积核 为1X1
        self.branch1X1 = nn.Conv2d(in_channels,16,kernel_size=1)
        # 二维的卷积层 输入 inchannel 通道 输入16个通道 卷积核为1X1
        self.branch5X5_1 = nn.Conv2d(in_channels,16,kernel_size=1)
        #二维的卷积层 输入16通道 输出24通道 卷积核 为1*1
        self.branch5X5_2 = nn.Conv2d(16,24,kernel_size=1)
        #二维的卷积层 输入inchannel 通道 输出16 卷积核为1*1
        self.branch3X3_1 = nn.Conv2d(in_channels,16,kernel_size=1)
        # 二维的卷积层 输入16 通道 输出24 卷积核为3X3 补边为1
        self.branch3X3_2 = nn.Conv2d(16,24,kernel_size=3,padding=1)
        self.branch3X3_3 = nn.Conv2d(24,24,kernel_size=3,padding=1)
        # 二维的卷积层 输入inchannel通道 输出24 卷积核为1*1
        self.branch_pool = nn.Conv2d(in_channels,24,kernel_size=1)
    def forward(self,x):
        """
        开始进行计算 每个都是不同的分支的计算 每个进行相关的池化操作或者是卷积操作最后进行拼接到一起
        :param x:
        :return:
        """

        branch1X1 = self.branch1X1(x)
        branch5X5 = self.branch5X5_1(x)
        branch5X5 = self.branch5X5_2(branch5X5)

        branch3X3 = self.branch3X3_1(x)
        branch3X3 = self.branch3X3_2(branch3X3)
        branch3X3 = self.branch3X3_3(branch3X3)

        branch_pool = F.avg_pool2d(x,kernel_sizeo=3,stride=1,padding = 1)
        #先拼接到一个数组里面 然后开始直接利用torch.cat方法来操作 dim 表示为方向
        output = [branch1X1,branch5X5,branch3X3,branch_pool]
        return torch.cat(output,dim=1)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        """
            利用上面的初始化操作 给出下面的直接进行使用
        """
        self.conv1 = nn.Conv2d(1,10,kernel_size=5)
        self.conv2 = nn.Conv2d(88,20,kernel_size=5)

        self.incep1 = InceptionA(in_channels=10)
        self.incep2 = InceptionA(in_channels=20)
        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Linear(1408,10)
    def forward(self,x):
        in_sieze = x.size(0)
        x = F.relu(self.mp(self.conv1(x)))

        x = self.incep1(x)

        x = F.relu(self.mp(self.conv2))
        x = self.incep2(x)

        x = x.view(in_sieze,-1)
        x = self.fc(x)
        return x