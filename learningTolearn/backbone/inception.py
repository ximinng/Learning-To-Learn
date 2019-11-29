# -*- coding: utf-8 -*-
"""
   Description : 
   Author :        xxm
"""
import torchvision

model = torchvision.models.inception_v3(pretrained=False)  # 我们不下载预训练权重
print(model)
