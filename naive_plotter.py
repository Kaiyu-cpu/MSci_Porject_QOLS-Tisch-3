# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 15:06:21 2023

@author: chaof
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv('Bayesian record.csv')

x = df.iloc[0][1:]
plt.plot(x)