# -*- coding: utf-8 -*-
"""
   Description : 
   Author :        xxm
"""

import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
print(curPath)

rootPath = os.path.split(curPath)[0]
print(rootPath)

sys.path.append(os.path.split(rootPath)[0])

print(
    sys.path
)
