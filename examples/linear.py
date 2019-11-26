# -*- coding: utf-8 -*-
"""
   Description : 
   Author :        xxm
"""
import torch
from torch.nn import Linear, Module, MSELoss
from torch.optim import SGD
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns

x = np.linspace(0, 20, 500)
y = 5 * x + 7
plt.plot(x, y)
