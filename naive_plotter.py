# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 15:06:21 2023

@author: chaof
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv('pygad record.csv')



Big_list = []

for i in range(8):
    x = df.iloc[i][1:]
    iter_list = []
    for j in range(0,len(x),8):
        sub_section = x[j:j+8]
        iter_list.append(max(sub_section))
            
    plt.plot(iter_list)
    Big_list.append(iter_list)
plt.show()    
mean_list = []

for j in range(len(Big_list[0])):
    sum_ = 0.0
    for i in range(8):
        sum_ += Big_list[i][j]
        
    mean_list.append(sum_/8)
    
plt.plot(mean_list)