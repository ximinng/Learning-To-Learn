# -*- coding: utf-8 -*-
"""
   Description :
   Author :        xxm
"""

import torch.nn as nn


def conv3x3(in_channels, out_channels, **kwargs):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, **kwargs),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


class PrototypicalNetwork(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_size=64):
        super(PrototypicalNetwork, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size

        self.encoder = nn.Sequential(
            conv3x3(in_channels, hidden_size),
            conv3x3(hidden_size, hidden_size),
            conv3x3(hidden_size, hidden_size),
            conv3x3(hidden_size, out_channels)
        )

    def forward(self, inputs):
        # mate-task
        # [batch, task, channel, width, height] 5-way-5-shot : 25 pictures
        # inputs: [16, 25, 1, 28, 28]
        # inputs.shape[2:] : [1, 28, 28]
        print(inputs.view(-1, *inputs.shape[2:]).shape)

        # Train before embedding [1200, 1, 28, 28]
        # Train before embedding [400, 1, 28, 28]
        embeddings = self.encoder(inputs.view(-1, *inputs.shape[2:]))

        # Train after embedding: [400, 64, 1, 1]
        # Test after embedding: [1200, 64, 1, 1]
        return embeddings.view(*inputs.shape[:2], -1)
