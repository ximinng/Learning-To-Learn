# -*- coding: utf-8 -*-
"""
   Description :   合并与分割
   Author :        xxm
"""
# In[]:
import torch

# In[]: cat合并 -- 需要除了cat之外的dim一致
a = torch.rand(4, 32, 8)
b = torch.rand(5, 32, 8)

torch.cat([a, b], dim=0)  # dim指定合并的维度
# Out[5]: torch.Size([9, 32, 8])

a1 = torch.rand(4, 3, 16, 32)
a2 = torch.rand(4, 3, 16, 32)
torch.cat([a1, a2], dim=2).shape
# Out[7]: torch.Size([4, 3, 32, 32])

# In[]: stack 合并并创建一个新的维度
# 在dim指定的位置上创建新的维度,两个tensor的dim指定的维度必须一致

torch.stack([a1, a2], dim=2).shape
# Out[8]: torch.Size([4, 3, 2, 16, 32])

a3 = torch.rand(32, 8)
a4 = torch.rand(32, 8)
a_stack = torch.stack([a3, a4], dim=0)
print(a_stack.shape)
# Out[9]: torch.Size([2, 32, 8])

# In[]: split根据区间长度拆分

c1, c2 = a_stack.split(split_size=1, dim=0)  # 第0个维度,每个区间长度为1
print(c1.shape)
# Out[8]: torch.Size([1, 32, 8])
print(c2.shape)
# Out[9]: torch.Size([1, 32, 8])

s1, s2, s3 = a_stack.split([16, 8, 8], dim=1)  # 给出区间等分规则进行拆分
print(s1.shape)
# Out[13]: torch.Size([2, 16, 8])
print(s2.shape)
# Out[14]: torch.Size([2, 8, 8])
print(s3.shape)
# Out[15]: torch.Size([2, 8, 8])

# In[]: chunk按数量拆分
aa, bb = a_stack.chunk(chunks=2, dim=0)  # 将第0个维度拆分为两个部分
print(aa.shape)
# Out[8]: torch.Size([1, 32, 8])
print(bb.shape)
# Out[9]: torch.Size([1, 32, 8])
