# -*- coding: utf-8 -*-
"""
   Description :
   Author :        xxm
"""
import os
import sys

path = {
    'BASE_PATH': os.path.abspath(os.path.dirname(os.path.dirname(__file__))),
    'LEARNING_TO_LEARN': os.path.dirname(os.path.abspath(__file__))
}
for k, v in path.items():
    print(v)
    sys.path.append(v)

print(sys.path)

# from learningTolearn import backbone, dataloader, method, test, train, util
#
# __all__ = ['backbone', 'dataloader', 'method', 'test', 'train', 'util']
