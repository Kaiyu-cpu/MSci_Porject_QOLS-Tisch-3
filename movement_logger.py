# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 21:51:31 2023

@author: chaof
"""

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0,150,1501)

def V_to_rad(V):
    rad = V/150*525  #units micro-rad
    return rad

y = V_to_rad(x)

plt.plot(x,y)
plt.xlabel('voltage (V)')
plt.ylabel('angular deflection (murad)')
plt.grid()