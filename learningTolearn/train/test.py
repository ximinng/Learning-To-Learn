# -*- coding: utf-8 -*-
"""
   Description : 
   Author :        xxm
"""

import os

folder = '/Data/'

folder = os.path.abspath(folder)
print(folder)
model_path = os.path.abspath(os.path.join(folder, 'model.th'))
print(model_path)