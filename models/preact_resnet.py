"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

# based on https://github.com/kuangliu/pytorch-cifar/blob/master/models/preact_resnet.py

'''Pre-activation ResNet in PyTorch.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.gate_layer import GateLayer


def norm2d(planes, num_groups=32):
    if num_groups != 0:
        print("num_groups:{}".format(num_groups))
    if num_groups > 0:
        return GroupNorm2D(planes, num_groups)
    else:
        return nn.BatchNorm2d(planes)


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, group_norm=0):
        super(PreActBlock, self).__init__()
        self.bn1 = norm2d(in_planes, group_norm)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gate1 = GateLayer(planes,planes,[1, -1, 1, 1])
        self.bn2 = norm2d(planes, group_norm)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.gate_out = GateLayer(planes,planes,[1, -1, 1, 1])

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )
            self.gate_shortcut = GateLayer(self.expansion*planes,self.expansion*planes,[1, -1, 1, 1])

    def forward(self, x):
        out = F.relu(self.bn1(x))

        if hasattr(self, 'shortcut'):
            shortcut = self.shortcut(out)
            shortcut = self.gate_shortcut(shortcut)
        else:
            shortcut = x

        out = self.conv1(out)

        out = self.bn2(out)
        out = self.gate1(out)

        out = F.relu(out)
        out = self.conv2(out)
        out = self.gate_out(out)

        out = out + shortcut
        ##as a block here we might benefit with gate at this stage

        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, group_norm=0):
        super(PreActBottleneck, self).__init__()

        self.bn1 = norm2d(in_planes, group_norm)

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = norm2d(planes, group_norm)
        self.gate1 = GateLayer(planes,planes,[1, -1, 1, 1])

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = norm2d(planes, group_norm)
        self.gate2 = GateLayer(planes,planes,[1, -1, 1, 1])

        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.gate3 = GateLayer(self.expansion*planes,self.expansion*planes,[1, -1, 1, 1])

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False))
            self.gate_shortcut = GateLayer(self.expansion*planes,self.expansion*planes,[1, -1, 1, 1])

    def forward(self, x):
        out = F.relu(self.bn1(x))
        input_out = out

        out = self.conv1(out)
        out = self.bn2(out)
        out = self.gate1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn3(out)
        out = self.gate2(out)

        out = F.relu(out)

        out = self.conv3(out)
        out = self.gate3(out)

        if hasattr(self, 'shortcut'):
            shortcut = self.shortcut(input_out)
            shortcut = self.gate_shortcut(shortcut)
        else:
            shortcut = x

        out = out + shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, group_norm=0, dataset="CIFAR10"):
        super(PreActResNet, self).__init__()

        self.in_planes = 64
        self.dataset = dataset

        if dataset == "CIFAR10":
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            num_classes = 10
        elif dataset == "Imagenet":
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                           bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            num_classes = 1000

        self.gate_in = GateLayer(64, 64, [1, -1, 1, 1])

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, group_norm=group_norm)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, group_norm=group_norm)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, group_norm=group_norm)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, group_norm=group_norm)

        if dataset == "CIFAR10":
            self.avgpool = nn.AvgPool2d(4, stride=1)
            self.linear = nn.Linear(512*block.expansion, num_classes)
        elif dataset == "Imagenet":
            self.avgpool = nn.AvgPool2d(7, stride=1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)



        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.BatchNorm2d):
                m.bias.data.zero_()

    def _make_layer(self, block, planes, num_blocks, stride, group_norm = 0):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, group_norm = group_norm))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        if self.dataset == "Imagenet":
            out = self.bn1(out)
            out = self.relu(out)
            out = self.maxpool(out)

        out = self.gate_in(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)

        out = out.view(out.size(0), -1)
        if self.dataset == "CIFAR10":
            out = self.linear(out)
        elif self.dataset == "Imagenet":
            out = self.fc(out)

        return out


def PreActResNet18(group_norm = 0):
    return PreActResNet(PreActBlock, [2,2,2,2], group_norm= group_norm)

def PreActResNet34():
    return PreActResNet(PreActBlock, [3,4,6,3])

def PreActResNet50(group_norm=0, dataset = "CIFAR10"):
    return PreActResNet(PreActBottleneck, [3,4,6,3], group_norm = group_norm, dataset = dataset)

def PreActResNet101():
    return PreActResNet(PreActBottleneck, [3,4,23,3])

def PreActResNet152():
    return PreActResNet(PreActBottleneck, [3,8,36,3])


def test():
    net = PreActResNet18()
    y = net((torch.randn(1,3,32,32)))
    print(y.size())

# test()
