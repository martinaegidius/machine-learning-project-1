#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 15:40:58 2022

@author: group 
"""
# %%
# Load packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import svd

# %%
# Load data and remove rows with NA
data = pd.read_csv('../Data/marketing.data', sep=",", header=None)[0].tolist() 

for i in range(len(data)):
    data[i] = data[i].split()

i = 0;

found = 0;
    
while(i < len(data)):
    for j in range(len(data[i])):
        if (data[i][j] == 'NA'):
            data.pop(i)
            found = 1;
            break
        else:
            found = 0;
    if (found != 1):
        i += 1;
    
for i in range(len(data)):
    for j in range(len(data[i])):
        data[i][j] = int(data[i][j])
        

# %%
# Data to numpy array

X = np.array(data) # Data
N = np.shape(data)[0] # Number of data objects
M = np.shape(data)[1] # Number of attributes
attributeNames = np.array(['Annual income of household', 
                           'Sex', 
                           'Maritial status', 
                           'Age',
                           'Education',
                           'Occupation',
                           'Time lived in the San Fran./Oakland/San Jose area',
                           'Dual incomes (if married)',
                           'Persons in your household',
                           'Persons in your household under 18',
                           'Householder status',
                           'Type of home',
                           'Ethnic Classification',
                           'Language spoken most often in home'])

annualIncomeMarker = ['Less than \$10,000',
                      '\$10,000 to \$14,999',
                      '\$15,000 to \$19,999',
                      '\$20,000 to \$24,999',
                      '\$25,000 to \$29,999',
                      '\$30,000 to \$39,999',
                      '\$40,000 to \$49,999',
                      '\$50,000 to \$74,999',
                      '\$75,000 or more']


#



# %%
# Principal Component Analysis

# Subtract mean value from data

Y = X - np.ones((N,1))*X.mean(axis=0)

# PCA by computing SVD of Y
U,S,V = svd(Y,full_matrices=False)


# Compute variance explained by principal components
rho = (S*S) / (S*S).sum() 

threshold = 0.9

# Plot variance explained
plt.figure()
plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold, threshold],'k--')
plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()
plt.show()


#%%
i=0
plt.figure()
plt.hist(X[:,i],alpha=1,bins = 9,rwidth=0.7, orientation="horizontal")
plt.title(attributeNames[i])
plt.yticks([1.5,2.5,3,4,5,6,7,8,9],annualIncomeMarker,fontsize=10)
plt.show()



#%% analyzing Principal components
Vp = V.T








