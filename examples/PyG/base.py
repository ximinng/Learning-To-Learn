# -*- coding: utf-8 -*-
"""
   Description : 
   Author :        xxm
"""
import torch
from torch_geometric.data import Data

edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)
x = torch.tensor([[-1], [0], [1]], dtype=torch.long)

data = Data(x=x, edge_index=edge_index)
# out: Data(edge_index=[2, 4], x=[3, 1])
# edge_index=[2, 4] : graph has 4/2 undirected edges
# x =[3, 1] : three nodes and each one having 1 feature
