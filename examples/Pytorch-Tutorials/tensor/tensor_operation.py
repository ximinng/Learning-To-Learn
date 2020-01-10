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

# matmul支持高维运算 -- 支持多个矩阵对并行相乘
c = torch.rand(4, 3, 28, 64)  # batch, channel, width, height
d = torch.rand(4, 3, 64, 32)
torch.matmul(c, d).shape  # matmul last two dimensions
# Out[5]: torch.Size([4, 3, 28, 32])

e = torch.rand(4, 1, 64, 32)  # matmul support broadcasting, channel 1->3
torch.matmul(c, e).shape  # matmul last two dimensions
# Out[5]: torch.Size([4, 3, 28, 32])

# In[]: 开方

aa = torch.full(size=[2, 2], fill_value=3)
# Out[5]: tensor([[3., 3.], [3., 3.]])

aa.pow(2)  # 开方
# Out[8]: tensor([[9., 9.], [9., 9.]])

aa.sqrt()  # 根号2
# Out[10]: tensor([[3., 3.], [3., 3.]])

# In[]: exp v.s. log

bb = torch.exp(torch.ones(2, 2))  # e^1 = e
# Out[15]: tensor([[2.7183, 2.7183], [2.7183, 2.7183]])

torch.log(bb)  # log(e) = 1
# Out[16]: tensor([[1., 1.], [1., 1.]])

# torch.log2()
# torch.log10()

# In[]: Approximation 近似

cc = torch.tensor(3.14)
print(cc.floor(),  # 向下取整
      cc.ceil(),  # 向上取整
      cc.trunc(),  # 整数部分
      cc.frac())  # 小数部分
# tensor(3.) tensor(4.) tensor(3.) tensor(0.1400)

print(torch.round(cc))  # 四舍五入取整
# tensor(3.)

# In[]: clamp 梯度裁剪
grad = torch.rand(2, 3) * 15
# Out[24]: tensor([[ 6.3465,  9.7351,  3.0044], [ 0.8178, 14.0766,  4.9276]])

print(grad.max())  # tensor(14.0766)
print(grad.min())  # tensor(0.8178)
print(grad.median())  # tensor(4.9276)

grad.clamp(6)
grad.clamp(min=0, max=10)  # 将梯度限制在 0~10 之间

# In[]: where 类似三元操作符 ? :

cond = torch.rand(2, 2)
# Out[25]: tensor([[0.6343, 0.7824], [0.3671, 0.7832]])
a = torch.zeros(2, 2)
b = torch.ones(2, 2)

torch.where(condition=cond > 0.5, input=a, other=b)
# Out[28]:tensor([[0., 0.], [1., 0.]])

# In[]: gather

prob = torch.randn(4, 10)

_, idx = prob.topk(dim=1, k=3)

label = torch.arange(10) + 100

torch.gather(label.expand(4, 10), dim=1, index=idx.long())
