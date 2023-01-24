#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 22:17:26 2023

@author: fanchao
"""

from bayes_opt import BayesianOptimization
import numpy as np
import matplotlib.pyplot as plt
# Bounded region of parameter space
pbounds = {'x':(0,75),'y':(0,75),'z':(0,75),'w':(0,75)}


def linear(x,y,z,w):
    
    return x+y+z+w


def test_func(x1,x2):
    return (np.sin(np.pi*x1/86-2*np.pi*x2/51)**2+np.sin(np.pi*x2/102+2*np.pi*x1/43)**2)/(np.sqrt((x1-43)**2+(x2-51)**2)/10+1)

def non_linear(x,y,z,w):
    return -x-y+test_func(z,w)

optimizer = BayesianOptimization(
    f=linear,
    pbounds=pbounds,
    random_state=1,
    allow_duplicate_points=True)


optimizer.maximize(init_points=1,n_iter=30)

plt.plot(range(1, 1 + len(optimizer.space.target)), optimizer.space.target, "-o")