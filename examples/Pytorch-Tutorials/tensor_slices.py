# -*- coding: utf-8 -*-
"""
   Description :   切片与索引
   Author :        xxm
"""
import torch

# In[]: select first/last N

a = torch.rand(4, 3, 28, 28)  # pic, channel, width, height
# a[0].shape
# Out[17]: torch.Size([3, 28, 28])

# a[:2].shape
# Out[19]: torch.Size([2, 3, 28, 28])

# a[:2, :1, :, :].shape
# Out[22]: torch.Size([2, 1, 28, 28])

# a[:2, -1:, :, :].shape # 从后向前索引,最后一个元素索引为-1
# Out[3]: torch.Size([2, 1, 28, 28])

# a[:, :, 0:28:2, 0:28:2].shape # 隔行索引 0:28:2 from index 0 to 18 step 2 左闭右开
# Out[5]: torch.Size([4, 3, 14, 14])

# a[:, :, ::2, ::2].shape # ::2 from start to end step 2
# Out[6]: torch.Size([4, 3, 14, 14])

# In[]: select by specific index

# a.index_select(0, torch.tensor([0,2])).shape # (which dim, range:tensor)
# Out[9]: torch.Size([2, 3, 28, 28])

# a.index_select(2, torch.arange(28)).shape
# Out[12]: torch.Size([4, 3, 28, 28])

# In[]: ... means var dims

# a[..., :2].shape == a[:,:,:,:2].shape
# Out[14]: torch.Size([4, 3, 28, 2])

# In[]: select by mask

# 取出所有概率大于0.5的元素
x = torch.randn(3, 4)
mask = x.ge(0.5)  # >=
torch.masked_select(x, mask)

# In[]: select by flatten index 摊平

src = torch.tensor([[4, 3, 5], [6, 7, 8]])
torch.take(src, torch.tensor([0, 2, 5]))
