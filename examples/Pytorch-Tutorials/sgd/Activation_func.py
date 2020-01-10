# -*- coding: utf-8 -*-
"""
   Description :   激活函数
   Author :        xxm
"""

# In[]:
import torch
import torch.nn.functional as F

# In[]: sigmoid 易取得0与1

a = torch.linspace(-100, 100, 10)
"""
Out[4]: 
tensor([-100.0000,  -77.7778,  -55.5556,  -33.3333,  -11.1111,   11.1111,
          33.3333,   55.5555,   77.7778,  100.0000])
"""

torch.sigmoid(a)
"""
Out[3]: 
tensor([0.0000e+00, 1.6655e-34, 7.4564e-25, 3.3382e-15, 1.4945e-05, 9.9999e-01,
        1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00])
"""

# In[]: tanh(x) = 2*sigmoid(2x)-1

torch.tanh(a)

# In[]: ReLU

torch.relu(a)
"""
Out[3]: 
tensor([  0.0000,   0.0000,   0.0000,   0.0000,   0.0000,  11.1111,  33.3333,
         55.5555,  77.7778, 100.0000])
"""

# In[]: softmax: soft version of max
# 将原本大的scores概率放大，小的scores压缩更密集
"""
softmax` = pi*(1-pi) i==j
         = -pj*pi    i!=j
"""

b = torch.rand(3, requires_grad=True)
p = F.softmax(b, dim=0)
p[0].backward()  # 对loss反向传播求梯度
# tensor([2.])
