# -*- coding: utf-8 -*-
"""
   Description :   The data type of tensor
   Author :        xxm
"""
# In[]
import torch

"""Dim 0"""
a = torch.tensor(1)  # scalar like loss value

"""Dim 1"""
b = torch.tensor([1, 1])  # Bias or Linear input

"""Dim 2"""
c = torch.randn(2, 784)  # (input batch, Linear input)

"""Dim 3"""
d = torch.rand(2, 2, 3)  # RNN (words, sentences, vector of words)

"""Dim 4"""
e = torch.rand(2, 3, 28, 28)  # CNN(batch size, channel, height, width) likes (second pic, channel, height, width)

# In[]:
"""create tensor from numpy"""
import numpy as np

np_a = np.array([2, 3, 4.4])
t_a = torch.from_numpy(np_a)

# In[]:
"""
Create tensor:
              tensor() accept value
              Tensor() accept shape
"""

torch.tensor([2., 3.2])  # accept value of tensor
point = torch.Tensor(2, 3, 3, 4)  # accept shape of tensor

print(point.shape)
print(point.dtype)

# In[]:
"""uninitialized"""
emptys = torch.empty(2, 3)  # 数据容器不要直接使用 数据差异性过大

# In[]:
"""Set Default Tensor Type"""
torch.set_default_tensor_type(torch.FloatTensor)
# torch.FloatTensor

# In[]:
"""init Tensor"""
rand = torch.rand(3, 3)

rand_like = torch.rand_like(rand)

torch.randint(low=1, high=10, size=[3, 3])

randInt = rand.to(dtype=torch.short)

# 填充
torch.full([], 7)
torch.full([2, 3], 7)

# 等差数列
torch.arange(0, 10, step=2)
# Out[11]: tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# 等分数列
torch.linspace(0, 10, steps=4)
# Out[12]: tensor([ 0.0000,  3.3333,  6.6667, 10.0000])
torch.logspace(0, -1, steps=10)  # 带有底数
# Out[13]: tensor([1.0000, 0.7743, 0.5995, 0.4642, 0.3594, 0.2783, 0.2154, 0.1668, 0.1292, 0.1000])

torch.ones(3, 3)
torch.zeros(3, 3)
torch.eye(3, 3)  # 单位矩阵

# In[]:

# 随机shuffle
torch.randperm(10)
# Out[14]: tensor([6, 5, 1, 3, 9, 4, 8, 7, 0, 2])

