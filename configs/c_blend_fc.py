#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
created on Wen. Feb 28th, 2018
                  by Xiaolong
"""
import torch
from torch import nn
import torch.nn.functional as F
# We use a simple densely-connected net for feature blending, without conv layers


class blendNet(nn.Module):
    def __init__(self):
        super(blendNet, self).__init__()
        self.relu = nn.LeakyReLU(0.01)
        self.dropout= nn.Dropout(p=0.5)
        self.conv1d = nn.Conv1d(1, 32, 2)# input channel, and output channel, kernel size
        self.fc1 = nn.Linear(8192, 256)
        self.fc2 = nn.Linear(256, 32)
        self.fc3 = nn.Linear(32, 5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return F.softmax(x, dim=0)


if __name__ == "__main__":
    dd = torch.randn(20, 1, 1, 4096)
    dd = torch.autograd.Variable(dd)
    model = blendNet()
    y = model(dd)

    print(y.size())




