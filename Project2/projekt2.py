#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 15:23:53 2022

@author: nikolinerehn
"""

# %% Load packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import svd
import matplotlib.patches as mpatches
from sklearn import tree
from platform import system
from os import getcwd
from toolbox_02450 import windows_graphviz_call
from matplotlib.image import imread
plt.style.use('seaborn')

# %% Load data and remove rows with NA
data = pd.read_csv('marketing.data.txt', sep=",", header=None)[0].tolist() 
#data = np.genfromtxt('../Project1/Data/marketing.data', delimiter=None) 
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
        

# %% Data to numpy array

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
                                             'Living together,\nnot married',
                                             'Divorced or\n separated', 
                                             'Widowed',
                                             'Single,\nnever married'] , 
              'Age':                        ['14 thru 17','18 thru 24', '25 thru 34','35 thru 44','45 thru 54', '55 thru 64','65 and Over'],
              'Education':                  ['Grade 8\nor less','Grades 9 to 11','Graduated \nhigh school','1 to 3 years\n of college','College \ngraduate','Grad Study'],
              'Occupation':                 ['Professional/\nManagerial','Sales Worker','Factory Worker/\nLaborer/Driver','Clerical/\nService Worker','Homemaker','Student, HS\n or College','Military','Retired','Unemployed'],
              'Dual incomes (if married)':['Not Married','Yes','No'],
              'Persons in your household':['One','Two','Three','Four','Five','Six','Seven','Eight','Nine or more'],
              'Persons in your household under 18':['None','One','Two','Three','Four','Five','Six','Seven','Eight','Nine or more'],
              'Householder status':['Own','Rent','Live with \nParents/Family'],
              'Type of home':['House','Condominium','Apartment','Mobile Home','Other'],
              'Ethnic Classification':['American \nIndian','Asian','Black','East Indian','Hispanic', 'Pacific \nIslander', 'White','Other'],
              'Language spoken most often in home':['English','Spanish','Other']}

#'Time lived in the San Fran./Oakland/San Jose area',
 #'Time lived in the San Fran./Oakland/San Jose area':['Less than one year','One to three years','Four to six years','Seven to ten years','More than ten years'],


# %%

attributeNames = ['Sex', 
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
                  'Language spoken most often in home']

attributeValues = X[:,1:]

classIndices = X[:,0]-1

classNames = ['Less than \$10,000',
              '\$10,000 to \$14,999',
              '\$15,000 to \$19,999',
              '\$20,000 to \$24,999',
              '\$25,000 to \$29,999',
              '\$30,000 to \$39,999',
              '\$40,000 to \$49,999',
              '\$50,000 to \$74,999',
              '\$75,000 or more']

# Number data objects, attributes, and classes
dataObjectNum, attributNum = attributeValues.shape
classNum = len(classNames)

# Add offset attribute
X = np.concatenate((np.ones((X.shape[0],1)),X),1)
attributeNames = [u'Offset']+attributeNames
M = M+1

#define y (predictor)






