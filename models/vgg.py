import torch, math
import torch.nn as nn
from .layers import Bias1D, Bias2D

class VGG(nn.Module):
    def __init__(self, features, num_classes=1000):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 49, 4096, bias=False), nn.ReLU(True), nn.Dropout(),
            Bias1D(4096), nn.Linear(4096, 4096, bias=False), nn.ReLU(True), nn.Dropout(),
            Bias1D(4096), nn.Linear(4096, num_classes))

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def make_layers(cfg):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            if len(layers) != 0:
                layers.append(Bias2D(in_channels))
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
            layers.append(conv2d)
            layers.append(nn.ReLU(inplace=True))
            in_channels = v
    layers.append(Bias2D(cfg[-2]))
    return nn.Sequential(*layers)


cfg16 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
cfg19 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']


def vgg16(num_classes=1000):
    return  VGG(make_layers(cfg16), num_classes=num_classes)

def vgg19(num_classes=1000):
    return  VGG(make_layers(cfg19), num_classes=num_classes)
