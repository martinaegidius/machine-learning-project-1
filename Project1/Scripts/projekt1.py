#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 15:40:58 2022

@author: nikolinerehn
"""
# %%
# Load packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import svd
import matplotlib.patches as mpatches
plt.style.use('seaborn')
# %%
# Load data and remove rows with NA
#data = pd.read_csv('marketing.data.txt', sep=",", header=None)[0].tolist() 
data = pd.read_csv('../Data/marketing.data', sep=",", header=None)[0].tolist() 

for i in range(len(data)):
    data[i] = data[i].split()

for i in range(len(data)):
    data[i].pop(6)

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
                           'Dual incomes (if married)',
                           'Persons in your household',
                           'Persons in your household under 18',
                           'Householder status',
                           'Type of home',
                           'Ethnic Classification',
                           'Language spoken most often in home'])

yTicksDict = {'Annual income of household': ['Less than \$10,000',
                                             '\$10,000 to \$14,999',
                                             '\$15,000 to \$19,999',
                                             '\$20,000 to \$24,999',
                                             '\$25,000 to \$29,999',
                                             '\$30,000 to \$39,999',
                                             '\$40,000 to \$49,999',
                                             '\$50,000 to \$74,999',
                                             '\$75,000 or more'], 
              'Sex':                        ['Male','Female'], 
              'Maritial status':            ['Married',
                                             'Living together, not married',
                                             'Divorced or separated', 
                                             'Widowed',
                                             'Single, never married'] , 
              'Age':                        ['14 thru 17','18 thru 24', '25 thru 34','35 thru 44','45 thru 54', '55 thru 64','65 and Over'],
              'Education':                  ['Grade 8 or less','Grades 9 to 11','Graduated high school','1 to 3 years of college','College graduate','Grad Study'],
              'Occupation':                 ['Professional/Managerial','Sales Worker','Factory Worker/Laborer/Driver','Clerical/Service Worker','Homemaker','Student, HS or College','Military','Retired','Unemployed'],
              'Dual incomes (if married)':['Not Married','Yes','No'],
              'Persons in your household':['One','Two','Three','Four','Five','Six','Seven','Eight','Nine or more'],
              'Persons in your household under 18':['None','One','Two','Three','Four','Five','Six','Seven','Eight','Nine or more'],
              'Householder status':['Own','Rent','Live with Parents/Family'],
              'Type of home':['House','Condominium','Apartment','Mobile Home','Other'],
              'Ethnic Classification':['American Indian','Asian','Black','East Indian','Hispanic', 'Pacific Islander', 'White','Other'],
              'Language spoken most often in home':['English','Spanish','Other']}

#'Time lived in the San Fran./Oakland/San Jose area',
 #'Time lived in the San Fran./Oakland/San Jose area':['Less than one year','One to three years','Four to six years','Seven to ten years','More than ten years'],

# %%
# Principal Component Analysis

# Subtract mean value from data
Y = X - np.ones((N,1))*X.mean(axis=0)

# PCA by computing SVD of Y
U,S,Vh = svd(Y,full_matrices=False)

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
plt.grid(True)
plt.show()






# %% Scatter plots

np.random.seed(2)
scatterData = np.copy(X).astype(np.float64)

for i in range(M):
    scatterArray = np.random.normal(0,0.3,N)
    scatterData[:,i] += scatterArray
    
    
# %%

colormap = np.array([ 'tab:brown','tab:olive','tab:orange','tab:red','tab:pink','tab:purple','tab:blue','tab:cyan','tab:green'])

leg = []

for i in range(9):
    leg.append(mpatches.Patch(color=colormap[i], label=yTicksDict[attributeNames[0]][i]))
    
# %%

plt.style.use('seaborn')

for i in range(M):
    for j in range(i+1,M):
        
        plt.figure()
        h = plt.scatter(scatterData[:,i],scatterData[:,j],s=0.7,c=colormap[X[:,0]-1])
        plt.xlabel(attributeNames[i])
        if j == 8:
            plt.yticks(range(len(yTicksDict[attributeNames[j]])),yTicksDict[attributeNames[j]],fontsize=7)
        else:
            plt.yticks(range(1,1+len(yTicksDict[attributeNames[j]])),yTicksDict[attributeNames[j]],fontsize=7)
        if i == 8:
            plt.xticks(range(len(yTicksDict[attributeNames[i]])),yTicksDict[attributeNames[i]],fontsize=7)
        else:
            plt.xticks(range(1,1+len(yTicksDict[attributeNames[i]])),yTicksDict[attributeNames[i]],fontsize=7)
        ax = plt.gca()
        ax.set_xticklabels(labels=yTicksDict[attributeNames[i]],rotation=90,fontsize=7);
        plt.ylabel(attributeNames[j])
        plt.legend(handles=leg,scatterpoints=1,
           loc='lower left',
           ncol=3,
           fontsize=8)
        plt.show()














