# -*- coding: utf-8 -*-
"""
   Description :
   Author :        xxm
"""
import os
import sys

path = {
    'BASE_PATH': os.path.abspath(os.path.dirname(os.path.dirname(__file__))),
}

current_directory = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.dirname(current_directory) + os.path.sep + ".")
sys.path.append(root_path)

from learningTolearn import backbone, dataloader, method, test, train, util

__all__ = ['backbone', 'dataloader', 'method', 'test', 'train', 'util']
