# -*- coding: utf-8 -*-
"""
   Description :   操作维度
   Author :        xxm
"""

# In[]:
import torch

# In[]: view = reshape : 维度变换

a = torch.rand(4, 1, 28, 28)
print(a.shape)
print(a.view(4, 28 * 28))  # [4, 28x28] -> [4, 784]

# a.view(4, 28*28).shape
# Out[3]: torch.Size([4, 784])

# In[]: Squeeze v.s. unsqueeze 新增维度

# a.shape
# Out[3]: torch.Size([4, 1, 28, 28])

# a.unsqueeze(0).shape
# Out[4]: torch.Size([1, 4, 1, 28, 28])
# a.unsqueeze(-1).shape
# Out[5]: torch.Size([4, 1, 28, 28, 1])

# 可增加的范围: [-a.dim() -1, a.dim()+1) like [-5, 5)
# 正索引:from start 负索引:from end

# -------------------#-------------------
# For example: bias + feature_map

bias = torch.rand(32)
# bias.size()
# Out[11]: torch.Size([32])
feature_map = torch.rand(4, 32, 14, 14)

bias = bias.unsqueeze(1).unsqueeze(2).unsqueeze(0)  # 为bias增加维度
# bias.shape
# Out[13]: torch.Size([1, 32, 1, 1])

# In[]: Squeeze 删减维度 v.s. unsqueeze

# bias.shape
# Out[16]: torch.Size([1, 32, 1, 1])
# bias.squeeze(0).shape
# Out[17]: torch.Size([32, 1, 1])

# 可删减的范围: [-a.dim() -1, a.dim()+1) like [-5, 5)
# 正索引:from start 负索引:from end

# In[]: Expand 维度扩展 v.s. repeat
# Expand: broadcasting
# Repeat: memory copies

# bias.shape
# Out[16]: torch.Size([1, 32, 1, 1])
# bias.expand(4, 32, 14, 14).shape
# Out[21]: torch.Size([4, 32, 14, 14])

# 保持某一维度不变: -1
# bias.expand(-1, 32, 14, 14).shape
# Out[25]: torch.Size([1, 32, 14, 14])

# In[]: Expand v.s. repeat 维度扩展

# bias.shape
# Out[16]: torch.Size([1, 32, 1, 1])
# bias.repeat(4, 1, 14, 14).shape  # 这里4与14是指x4和x14
# Out[29]: torch.Size([4, 32, 14, 14])

# In[]: 转置 -- 仅2D

torch.tensor([[1, 2], [3, 4]])
# Out[38]: tensor([[1, 2], [3, 4]])

torch.tensor([[1, 2], [3, 4]]).t()
# Out[37]:tensor([[1, 3], [2, 4]])

# In[]: Transpose 维度交换 -- 每次仅交换一个维度
a = torch.randn(4, 3, 32, 32)  # torch.Size([4, 1, 28, 28])

# 矩阵transpose后数据内存不连续,contiguous()使其连续
# 注意view()函数丢失维度信息 -- 维度顺序信息丢失
# [b, c, H, W] -> [b, W, H, c] -> [b, c, W, H]
a1 = a.transpose(1, 3).contiguous().view(4, 3 * 32 * 32).view(4, 3, 32, 32)

# [b, c, H, W] -> [b, W, H, c] -> [b, W, H, c] -> [b ,c ,H, W]
a2 = a.transpose(1, 3).contiguous().view(4, 3 * 32 * 32).view(4, 32, 32, 3).transpose(1, 3)

print(torch.all(torch.eq(a, a1)))  # False
print(torch.all(torch.eq(a, a2)))  # True

# In[]: permute 维度交换 -- 交换多个维度

# 目标: [b, c, H, W] -> [b, H, W, c]

src = torch.rand(4, 3, 28, 32)
# src.shape
# Out[62]: torch.Size([4, 3, 28, 32])
tar = src.permute(0, 2, 3, 1)  # index of dims
# tar.shape
# Out[64]: torch.Size([4, 28, 32, 3])
