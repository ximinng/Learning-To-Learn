# -*- coding: utf-8 -*-
"""
   Description :   tensor的统计方法
   Author :        xxm
"""
# In[]: norm 范数
import torch

a = torch.full([8], 1)
b = a.view(2, 4)
c = a.view(2, 2, 2)

print(a.norm(1), b.norm(1), c.norm(1))
# tensor(8.) tensor(8.) tensor(8.)

print(a.norm(2), b.norm(2), c.norm(2))
# tensor(2.8284) tensor(2.8284) tensor(2.8284)

print(b.norm(1, dim=1))  # tensor([4., 4.])
print(b.norm(2, dim=1))  # tensor([2., 2.])

print(c.norm(1, dim=0))  # tensor([[2., 2.], [2., 2.]])
print(c.norm(2, dim=0))  # tensor([[1.4142, 1.4142], [1.4142, 1.4142]])

# In[]: 取整函数
d = torch.arange(8).view(2, 4).float()

print(d.min(),  # tensor(0.)
      d.max(),  # tensor(7.)
      d.mean(),  # tensor(3.5000) sum/size
      d.prod())  # tensor(0.)

print(d.sum())  # tensor(28.)

# index of max or min
print(d.argmax(), d.argmin())  # tensor(7) tensor(0)

aa = torch.rand(4, 10)
"""
Out[5]: 
tensor([[0.5173, 0.6426, 0.3534, 0.3929, 0.7328, 0.0805, 0.4122, 0.3430, 0.4223,
         0.2771],
        [0.8850, 0.9756, 0.2632, 0.5072, 0.2039, 0.0556, 0.5065, 0.2601, 0.6313,
         0.3372],
        [0.2598, 0.3514, 0.1471, 0.3618, 0.4594, 0.6788, 0.5365, 0.9053, 0.8129,
         0.6683],
        [0.1745, 0.6845, 0.5842, 0.0061, 0.8459, 0.9152, 0.6667, 0.2443, 0.9212,
         0.9137]])
"""

aa.max(dim=1)
"""
Out[6]: 
torch.return_types.max(
values=tensor([0.7328, 0.9756, 0.9053, 0.9212]),
indices=tensor([4, 1, 7, 8]))
"""

# In[]: Top k

aa.topk(3, dim=1)
"""
Out[13]: 
torch.return_types.topk(
values=tensor([[0.7328, 0.6426, 0.5173],
        [0.9756, 0.8850, 0.6313],
        [0.9053, 0.8129, 0.6788],
        [0.9212, 0.9152, 0.9137]]),
indices=tensor([[4, 1, 0],
        [1, 0, 8],
        [7, 8, 5],
        [8, 5, 9]]))
"""

# In[]:

# aa > 0
"""
Out[15]: 
tensor([[True, True, True, True, True, True, True, True, True, True],
        [True, True, True, True, True, True, True, True, True, True],
        [True, True, True, True, True, True, True, True, True, True],
        [True, True, True, True, True, True, True, True, True, True]])
"""

bb = torch.tensor([1, 2, 3])
# Out[18]: tensor([1, 2, 3])
cc = torch.arange(1, 4)
# Out[21]: tensor([1, 2, 3])

# eq() 返回每个元素是否相等
bb.eq(cc)
# Out[22]: tensor([True, True, True])

# equal() 仅返回整体是否相等的最终结果
bb.equal(cc)
# Out[23]: True
