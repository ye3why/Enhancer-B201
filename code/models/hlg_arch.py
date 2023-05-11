import torch
import torch.nn as nn
import torch.nn.functional as F
from registry import MODEL_REGISTRY


# 用于ResNet18和34的残差块，用的是2个3x3的卷积
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, dilation = 1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=1, padding=dilation, dilation=dilation, bias=False)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=dilation, dilation=dilation, bias=False)
        self.shortcut = nn.Sequential()
        # 经过处理后的x要与x的维度相同(尺寸和深度)
        # 如果不相同，需要添加卷积+BN来变换为同一维度
        if in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=1, bias=False),
            )
        self.mini1 = MiniBlock(planes, planes)
        self.mini2 = MiniBlock(planes, planes)
    def forward(self, x):
        out = F.relu((self.conv1(x)))
        out = self.mini1(out)
        out = self.mini2(out)
        out = (self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class MiniBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(MiniBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1,
                               stride=1, padding=0, bias=False)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=1,
                               stride=1, padding=0, bias=False)
        #self.shortcut = nn.Sequential()


    def forward(self, x):
        out = F.relu((self.conv1(x)))
        out = (self.conv2(out))
        out += x
        out = F.relu(out)
        return out

@MODEL_REGISTRY.register()
class LEBDE(nn.Module):
    expansion = 1

    def __init__(self, planes=32):
        super(LEBDE, self).__init__()
        self.conv1 = nn.Conv2d(3, planes, kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.block1 = BasicBlock(32,32,1)
        self.block2 = BasicBlock(32,32,1)
        self.block3 = BasicBlock(32,32,2)
        self.block4 = BasicBlock(32,32,2)
        self.block5 = BasicBlock(32,32,4)
        self.block6 = BasicBlock(32,32,4)
        self.block7 = BasicBlock(32,32,8)
        self.block8 = BasicBlock(32,32,8)
        self.conv2 = nn.Conv2d(planes, 3, kernel_size=1,
                               stride=1, padding=0, bias=False)
        #self.shortcut = nn.Sequential()

    def _forward(self, x):
        res = F.relu(self.conv1(x))
        res = (self.block1(res))
        res = (self.block2(res))
        res = (self.block3(res))
        res = (self.block4(res))
        res = (self.block5(res))
        res = (self.block6(res))
        res = (self.block7(res))
        res = (self.block8(res))
        res = self.conv2(res)
        out = res + x
        return out

    def forward(self, x):
        return self._forward(x * 1023)
