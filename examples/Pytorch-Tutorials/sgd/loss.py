# -*- coding: utf-8 -*-
"""
   Description : 
   Author :        xxm
"""
# In[]:
import torch
import torch.nn.functional as F

# In[]: 自动求导: torch.autograd.grad
x = torch.ones(1)
w = torch.full([1], 2, requires_grad=True)
# w.requires_grad_()

mse = F.mse_loss(torch.ones(1), x * w)  # pred , label
# Out[3]: tensor(1.)
torch.autograd.grad(mse, [w])
# Out[4]: (tensor([2.]),)

# In[]: 反向传播: 对最终的loss调用backward()
x = torch.ones(1)
w = torch.full([1], 2, requires_grad=True)
# w.requires_grad_()

mse = F.mse_loss(torch.ones(1), x * w)  # pred , label
# Out[3]: tensor(1.)
mse.backward()

print(w.grad)
# Out[3]: tensor([2.])
