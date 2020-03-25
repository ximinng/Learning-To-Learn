# -*- coding: utf-8 -*-
"""
   Description :
   Author :        xxm
"""
import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

path = {
    'BASE_PATH': os.path.abspath(os.path.dirname(os.path.dirname(__file__))),
    'LEARNING_TO_LEARN': os.path.dirname(os.path.abspath(__file__))
}

for k, v in path.items():
    sys.path.append(v)

#