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
from matplotlib.pyplot import figure, plot, title, xlabel, ylabel, show, legend
plt.style.use('seaborn')
# %%
# Load data and remove rows with NA
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
"""
plt.style.use('seaborn')

for i in range(M):
    for j in range(i+1,M):
        
        plt.figure()
        plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.2)
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
        plt.gcf().set_size_inches(8, 6)
        plt.savefig('scatter/img{}_{}.png'.format(i,j))
        plt.show()

"""

# %%
"""
for i in range(M):
    for j in range(i+1,M):
        print('\\begin{figure}[H]')
        print('\\centering')
        print('\\includegraphics[width=13cm]{{Billeder/scatter/img{}_{}.png}}'.format(i,j))
        print('\\caption{}')
        print('\\label{{fig:{}_{}}}'.format(i,j))
        print('\\end{figure}')

"""

# %% Changing to continous data 

attributeNamesCon = np.array(['Annual income of household', 
                              'Age',
                              'Education',
                              'Persons in your household',
                              'Persons in your household under 18'])


TicksDict = {'Annual income of household': ['Less than \$10,000',
                                             '\$10,000 to \$14,999',
                                             '\$15,000 to \$19,999',
                                             '\$20,000 to \$24,999',
                                             '\$25,000 to \$29,999',
                                             '\$30,000 to \$39,999',
                                             '\$40,000 to \$49,999',
                                             '\$50,000 to \$74,999'], 
              'Age':                        ['14 thru 17',
                                             '18 thru 24',
                                             '25 thru 34',
                                             '35 thru 44',
                                             '45 thru 54',
                                             '55 thru 64'],
              'Education':                  ['Grade 8\nor less',
                                             'Grades 9 to 11',
                                             'Graduated \nhigh school',
                                             '1 to 3 years\n of college',
                                             'College \ngraduate',
                                             'Grad Study'],
              'Persons in your household':['One','Two','Three','Four','Five','Six','Seven','Eight','Nine or more'],
              'Persons in your household under 18':['None','One','Two','Three','Four','Five','Six','Seven','Eight','Nine or more']}

conAttri = np.array([1,4,5,9,10])-1


conData = np.copy(X)[:,conAttri].astype(float)

i = 0;

found = 0;
    
while(i < len(conData)):
    if conData[i,0] == 9 or conData[i,1] == 7:
        conData = np.delete(conData, i ,axis=0)
        found = 1;
    else:
        found = 0;
    if (found != 1):
        i += 1;

annualIncome = conData[:,0].astype(float);
age = conData[:,1].astype(float);


for i in range(len(conData)):
    if annualIncome[i] == 1:
        annualIncome[i] = np.random.rand()*10000
    elif annualIncome[i] == 2:
        annualIncome[i] = 10000 + np.random.rand()*5000
    elif annualIncome[i] == 3:
        annualIncome[i] = 15000 + np.random.rand()*5000
    elif annualIncome[i] == 4:
        annualIncome[i] = 20000 + np.random.rand()*5000
    elif annualIncome[i] == 5:
        annualIncome[i] = 25000 + np.random.rand()*5000
    elif annualIncome[i] == 6:
        annualIncome[i] = 30000 + np.random.rand()*10000
    elif annualIncome[i] == 7:
        annualIncome[i] = 40000 + np.random.rand()*10000
    elif annualIncome[i] == 8:
        annualIncome[i] = 50000 + np.random.rand()*25000
    if age[i] == 1:
        age[i] = 14 + np.random.rand()*3
    elif age[i] == 2:
        age[i] = 18 + np.random.rand()*6
    elif age[i] == 3:
        age[i] = 25 + np.random.rand()*9
    elif age[i] == 4:
        age[i] = 35 + np.random.rand()*9
    elif age[i] == 5:
        age[i] = 45 + np.random.rand()*9
    elif age[i] == 6:
        age[i] = 55 + np.random.rand()*9
        


##PCA 
# %%
# Principal Component Analysis
sdevs = np.std(conData,0)
conNorm = conData*(1/sdevs)
N = len(conData)

# Subtract mean value from data
Y = conNorm - np.ones((N,1))*conNorm.mean(axis=0)


# PCA by computing SVD of Y
U,S,Vh = svd(Y,full_matrices=False)

V = Vh.T #transposing for proper eigenvectors
# Project the centered data onto principal component space
Z = Y @ V

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




# Indices of the principal components to be plotted
i = 0
j = 1

# Plot PCA of the data
f = figure()
title('Marketing Data: PCA')
#Z = array(Z)
for c in range(5):
    # select indices belonging to class c:
    class_mask = y==c
    plot(Z[class_mask,i], Z[class_mask,j], 'o', alpha=.5)
legend(classNames)
xlabel('PC{0}'.format(i+1))
ylabel('PC{0}'.format(j+1))

# Output result to screen
show()

print('Ran Exercise 2.1.4')
















