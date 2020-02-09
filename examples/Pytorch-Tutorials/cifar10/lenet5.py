# -*- coding: utf-8 -*-
"""
   Description : 
   Author :        xxm
"""
import torch
from torch import nn
from torch.nn import functional as F


class Lenet5(nn.Module):

    def __init__(self):
        super(Lenet5, self).__init__()

        self.conv_unit = nn.Sequential(
            # x: [b, 3, 32, 32] -> [b, 6, 28, 28]
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=0),
            # -> [b, 6, 14, 14]
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            # x: [b, 6, 14, 14] -> [b, 16, 10, 10]
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),
            # -> [b, 16, 5, 5]
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        )

        self.fc_unit = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        # [b, 3, 32, 32] -> [b, 16, 5, 5]
        x = self.conv_unit(x)
        # [b, 16, 5, 5] -> [b, 16x5x5]
        x = x.view(x.size(0), 16 * 5 * 5)
        # [b, 16x5x5] -> [b, 10]
        logits = self.fc_unit(x)
        return logits
