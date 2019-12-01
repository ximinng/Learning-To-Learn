# -*- coding: utf-8 -*-
"""
   Description : 
   Author :        xxm
"""

import os
import sys

current_directory = os.path.dirname(os.path.abspath(__file__))
print(current_directory)
root_path = os.path.abspath(os.path.dirname(current_directory) + os.path.sep + ".")
print(root_path)
sys.path.append(root_path)

print(sys.path)
