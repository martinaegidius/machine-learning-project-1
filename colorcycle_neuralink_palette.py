#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 15:07:10 2022

@author: max
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# Set the default color cycle
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["#0B2436","#C9A3A4","#D1B899","#BBBBBE","#AEB5AC","#858686","#ADCDEC","#B6BDCA"]) 

x = np.linspace(0, 20, 100)

fig, axes = plt.subplots(nrows=2)

for i in range(10):
    axes[0].plot(x, i * (x - 10)**2)

for i in range(10):
    axes[1].plot(x, i * np.cos(x))

plt.show()