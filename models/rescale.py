import torch
import torch.nn as nn
from .layers import Bias2D

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, block_idx, max_block,
                 stride=1, groups=1, base_width=64, drop_conv=0.0):
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.)) * groups

        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, padding=1, stride=stride, groups=groups, bias=False)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, bias=False)

        self.addbias1 = Bias2D(inplanes)
        self.addbias2 = Bias2D(width)
        self.addbias3 = Bias2D(width)
        
        # self._scale1 = nn.Parameter(torch.ones(1, width, 1, 1))
        # self._scale2 = nn.Parameter(torch.ones(1, width, 1, 1))
        
        self._scale = nn.Parameter(torch.ones(1))

        multiplier = (block_idx + 1) ** -(1/6) * max_block **(1/6)
        multiplier = multiplier * (1 - drop_conv) ** .5

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                _, C, H, W = m.weight.shape
                stddev = (C * H * W / 2) ** -.5
                nn.init.normal_(m.weight, std = stddev * multiplier)

        self.residual = max_block ** -.5
        self.identity = block_idx ** .5 / (block_idx + 1) ** .5

        self.downsample = nn.Sequential()
        if stride != 1 or inplanes != self.expansion*planes:
            avgpool = nn.AvgPool2d(stride) if stride != 1 else nn.Sequential()
            self.downsample = nn.Sequential(
                avgpool, Bias2D(num_features=inplanes),
                nn.Conv2d(inplanes, self.expansion*planes, kernel_size=1, bias=False))
            nn.init.kaiming_normal_(self.downsample[2].weight, a=1)

        self.drop = nn.Sequential()
        if drop_conv > 0.0:
            self.drop = nn.Dropout2d(drop_conv)

    def forward(self, x):
        out = self.drop(self.conv1(self.addbias1(x))).relu_()
        # out = out.mul(self._scale1)
        out = self.drop(self.conv2(self.addbias2(out))).relu_()
        # out = out.mul(self._scale2)
        out = self.drop(self.conv3(self.addbias3(out)))

        out = out.mul(self._scale.mul(self.residual))
        out = torch.add(input=out, alpha=self.identity, other=self.downsample(x))
        return out.relu_()


class ReScale(nn.Module):

    def __init__(self, layers, num_classes=1000, groups=1, width_per_group=64, 
                 drop_conv=0.0, drop_fc=0.0):
        super(ReScale, self).__init__()
        block = Bottleneck

        self.inplanes = 64
        self.groups = groups
        self.base_width = width_per_group
        self.block_idx = sum(layers) - 1
        self.max_depth = sum(layers)

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.addbias1 = Bias2D(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, drop_conv=drop_conv)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, drop_conv=drop_conv)
        self.addbias2 = Bias2D(512*block.expansion)
        self.drop = nn.Dropout(drop_fc)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.fc.weight, a=1)

    def _make_layer(self, block, planes, num_blocks, stride=1, drop_conv=0.0):
        strides = [stride] + [1] * (num_blocks -1)
        layers = []
        for stride in strides:
            layers.append(block(self.inplanes, planes, block_idx=self.block_idx, max_block=self.max_depth, 
                stride=stride, groups=self.groups, base_width=self.base_width, drop_conv=drop_conv))
            self.inplanes = planes * block.expansion
            self.block_idx += 1
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.addbias1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.addbias2(x)
        x = x.mean((2,3))
        x = self.drop(x)
        x = self.fc(x)
        return x

def rescale50(num_classes=1000, drop_conv=0.0, drop_fc=0.0):
    return ReScale([3, 4, 6, 3], num_classes=num_classes,
        drop_conv=drop_conv, drop_fc=drop_fc, groups=1, width_per_group=64)

def rescale101(num_classes=1000, drop_conv=0.0, drop_fc=0.0):
    return ReScale([3, 4, 23, 3], num_classes=num_classes, 
        drop_conv=drop_conv, drop_fc=drop_fc, groups=1, width_per_group=64)

def rescale200(num_classes=1000, drop_conv=0.0, drop_fc=0.0):
    return ReScale([3,24,36,3], num_classes=num_classes,
        drop_conv=drop_conv, drop_fc=drop_fc, groups=1, width_per_group=64)

def rescaleX50_32x4d(num_classes=1000, drop_conv=0.0, drop_fc=0.0):
    return ReScale([3, 4, 6, 3], num_classes=num_classes, 
        drop_conv=drop_conv, drop_fc=drop_fc, groups=32, width_per_group=4)

def rescaleX101_32x8d(num_classes=1000, drop_conv=0.0, drop_fc=0.0):
    return ReScale([3, 4, 23, 3], num_classes=num_classes,
        drop_conv=drop_conv, drop_fc=drop_fc, groups=32, width_per_group=8)
