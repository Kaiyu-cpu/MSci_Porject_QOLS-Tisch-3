#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 11:51:06 2023

@author: fanchao
"""

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0,1,1000)
V = x
f = V - np.log10(1-V)
plt.plot(V,f)
plt.xlabel('Visibility')
plt.ylabel('scale function')
plt.grid()
