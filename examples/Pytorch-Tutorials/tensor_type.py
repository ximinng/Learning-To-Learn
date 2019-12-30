# -*- coding: utf-8 -*-
"""
   Description : 
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

# Out[]:
a

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

# In[]:
