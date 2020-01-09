# -*- coding: utf-8 -*-
"""
   Description :   tensor的运算
   Author :        xxm
"""
# In[]:
import torch

# In[]: 加减乘除
a = torch.ones(3, 4)
b = torch.tensor([2, 2, 2, 2])

# a + b
torch.add(a, b)
"""
Out[5]:
tensor([[3., 3., 3., 3.],
        [3., 3., 3., 3.],
        [3., 3., 3., 3.]])
"""

# a * b
torch.mul(a, b)
"""
Out[6]:
tensor([[2., 2., 2., 2.],
        [2., 2., 2., 2.],
        [2., 2., 2., 2.]])
"""

# a - b
torch.sub(a, b)
"""
Out[7]:
tensor([[-1., -1., -1., -1.],
        [-1., -1., -1., -1.],
        [-1., -1., -1., -1.]])
"""

# a / b
torch.div(a, b)
"""
Out[8]:
tensor([[0.5000, 0.5000, 0.5000, 0.5000],
        [0.5000, 0.5000, 0.5000, 0.5000],
        [0.5000, 0.5000, 0.5000, 0.5000]])
"""

# In[]: 矩阵乘法 mm(),matmul()和@ 但是mm仅限2d运算

# X@W+b : [4,784] -> [4,512]
# [4,784] @ [784,512] = [4,512]
X = torch.rand(4, 784)
w = torch.rand(512, 784)  # channel-out, channel-in
print(torch.matmul(X, w.t()).shape)
# Out[23]: torch.Size([4, 512])
