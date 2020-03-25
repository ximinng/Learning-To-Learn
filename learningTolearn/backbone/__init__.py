# -*- coding: utf-8 -*-
"""
   Description : 
   Author :        xxm
"""
from .conv import ModelConvMiniImagenet, ModelConvOmniglot
from .mlp import ModelMLPSinusoid
from .resnet import *

__all__ = [
    'ModelConvMiniImagenet', 'ModelConvOmniglot',
    'ModelMLPSinusoid',
    'resnet10', 'resnet12', 'resnet14', 'resnetbc14b', 'resnet16', 'resnet18_wd4', 'resnet18_wd2',
    'resnet18_w3d4', 'resnet18', 'resnet26', 'resnetbc26b', 'resnet34', 'resnetbc38b', 'resnet50', 'resnet50b',
    'resnet101', 'resnet101b', 'resnet152', 'resnet152b', 'resnet200', 'resnet200b'
]
