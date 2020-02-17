# -*- coding: utf-8 -*-
"""
   Description : 
   Author :        xxm
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

resnet18 = torchvision.models.resnet18(pretrained=False)  # 我们不下载预训练权重


class SkipConnection(nn.Module):
    def __init__(self, in_channels, out_channels, activation, batch_norm, stride):
        super(SkipConnection, self).__init__()
        self.batch_norm = batch_norm
        self.activation = activation

        if in_channels != out_channels or stride != 1:
            self.adjust = True
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
            if batch_norm:
                self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.adjust = False

    def forward(self, x):
        if self.adjust:
            out = self.conv(x)
            if self.batch_norm:
                out = self.bn(out)
            return self.activation(out)
        else:
            return x


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation, batch_norm, stride=1):
        super(BasicBlock, self).__init__()
        self.batch_norm = batch_norm
        self.activation = activation

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        if batch_norm:
            self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if batch_norm:
            self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = SkipConnection(in_channels, out_channels, activation, batch_norm, stride)

    def forward(self, x):
        out = self.conv1(x)
        if self.batch_norm:
            out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        if self.batch_norm:
            out = self.bn2(out)
        out = self.activation(out)

        out = out + self.shortcut(x)
        return out


class ResNet10(nn.Module):
    def __init__(self, block=BasicBlock, activation=F.relu, batch_norm=True, num_classes=10):
        super(ResNet10, self).__init__()
        self.activation = activation
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer = nn.Sequential(
            block(16, 16, activation, batch_norm, stride=1),
            block(16, 32, activation, batch_norm, stride=2),
            block(32, 64, activation, batch_norm, stride=2),
            block(64, 128, activation, batch_norm, stride=2)
        )
        self.linear = nn.Linear(128, num_classes)

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.layer(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation, batch_norm, stride=1):
        super(BottleneckBlock, self).__init__()
        self.batch_norm = batch_norm
        self.activation = activation
        half_in_c = in_channels // 2

        self.conv1 = nn.Conv2d(in_channels, half_in_c, kernel_size=1, stride=1, padding=0)
        if batch_norm:
            self.bn1 = nn.BatchNorm2d(half_in_c)
        self.conv2 = nn.Conv2d(half_in_c, half_in_c, kernel_size=3, stride=1, padding=1)
        if batch_norm:
            self.bn2 = nn.BatchNorm2d(half_in_c)
        self.conv3 = nn.Conv2d(half_in_c, out_channels, kernel_size=1, stride=stride, padding=0)
        if batch_norm:
            self.bn3 = nn.BatchNorm2d(out_channels)
        self.shortcut = SkipConnection(in_channels, out_channels, activation, batch_norm, stride)

    def forward(self, x):
        out = self.conv1(x)
        if self.batch_norm:
            out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        if self.batch_norm:
            out = self.bn2(out)
        out = self.activation(out)

        out = self.conv3(out)
        if self.batch_norm:
            out = self.bn3(out)
        out = self.activation(out)

        out = out + self.shortcut(x)
        return out
