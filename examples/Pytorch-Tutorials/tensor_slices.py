# -*- coding: utf-8 -*-
"""
   Description :
   Author :        xxm
"""
import torch

# In[]

a = torch.rand(4, 3, 28, 28)  # pic, channel, width, height
# a[0].shape
# Out[17]: torch.Size([3, 28, 28])

# a[:2].shape
# Out[19]: torch.Size([2, 3, 28, 28])

# a[:2, :1, :, :].shape
# Out[22]: torch.Size([2, 1, 28, 28])
