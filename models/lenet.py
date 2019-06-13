"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

# based on https://github.com/kuangliu/pytorch-cifar/blob/master/models/lenet.py

'''LeNet in PyTorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.gate_layer import GateLayer


class LeNet(nn.Module):
    def __init__(self, dataset="CIFAR10"):
        super(LeNet, self).__init__()
        if dataset=="CIFAR10":
            nunits_input = 3
            nuintis_fc = 32*5*5
        elif dataset=="MNIST":
            nunits_input = 1
            nuintis_fc = 32*4*4
        self.conv1 = nn.Conv2d(nunits_input, 16, 5)
        self.gate1 = GateLayer(16,16,[1, -1, 1, 1])
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.gate2 = GateLayer(32,32,[1, -1, 1, 1])
        self.fc1   = nn.Linear(nuintis_fc, 120)
        self.gate3 = GateLayer(120,120,[1, -1])
        self.fc2   = nn.Linear(120, 84)
        self.gate4 = GateLayer(84,84,[1, -1])
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = self.gate1(out)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = self.gate2(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.gate3(out)
        out = F.relu(self.fc2(out))
        out = self.gate4(out)
        out = self.fc3(out)
        return out