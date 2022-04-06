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
from toolbox_02450 import windows_graphviz_call,categoric2numeric
from matplotlib.image import imread
import sklearn.linear_model as lm
from sklearn.preprocessing import OneHotEncoder
plt.style.use('seaborn')
import sklearn.linear_model as lm
from sklearn import model_selection
from toolbox_02450 import rlr_validate
from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid)


# %% Load data and remove rows with NA
data = pd.read_csv('marketing.data.txt', sep=",", header=None)[0].tolist() 

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

fullData = np.copy(X);


# %% Duuno


# Split dataset into features and target vector
income_idx = np.where(attributeNames == 'Annual income of household')
annualIncome = X[:,0]

for i in range(len(annualIncome)):
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

y = annualIncome

#X_cols = list(range(0,income_idx)) + list(range(income_idx+1,len(attributeNames)))
X = X[:,1:]

# Fit ordinary least squares regression model
model = lm.LinearRegression()
model.fit(X,y)

# Predict alcohol content
y_est = model.predict(X)
residual = y_est-y

# Display scatter plot
plt.figure()
plt.subplot(2,1,1)
plt.plot(y, y_est, '.')
plt.xlabel('Income (true)'); plt.ylabel('Income (estimated)');
plt.subplot(2,1,2)
plt.hist(residual,40)

plt.show()


# %% Changing annual income to continous data

annualIncome = fullData[:,0]

for i in range(len(annualIncome)):
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
   




# %% One-out-of-K encoding

N = np.shape(fullData)[0] # Number of data objects
M = np.shape(fullData)[1] # Number of attributes

Kdata = {}

for i in range(M):
    K = categoric2numeric(fullData[:,i])
    K = (K[0],yTicksDict[attributeNames[i]])
    Kdata[i] = K
    
attributeValues_K = Kdata[1][0]
    
for i in range(2,M):
    attributeValues_K = np.hstack((attributeValues_K,Kdata[i][0]))
    
attributeValues_K = attributeValues_K.astype(np.float64)

# Normalization such that std = 1 and mean = 1

#row = N;
#col = np.shape(attributeValues_K)[1];


#for i in range(col):
#    attributeValues_K[:,i] = (attributeValues_K[:,i] - np.mean(attributeValues_K[:,i]))/np.std(attributeValues_K[:,i])

    
    

# %% 
classValues = fullData[:,0]


xlim = np.linspace(0,9,num=300)

# Fit ordinary least squares regression model
model = lm.LinearRegression()
model.fit(attributeValues_K,classValues)

# Predict alcohol content
y_est = model.predict(attributeValues_K)
residual = y_est-classValues
#ylim = model.predict(xlim)

# Display scatter plot
plt.figure()
plt.subplot(2,1,1)
plt.scatter(classValues+np.random.normal(size=N,scale=0.3), y_est,s=1)
#plt.plot(xlim,ylim,'r-')
plt.xlabel('Income (true)'); plt.ylabel('Income (estimated)');
plt.subplot(2,1,2)
plt.hist(residual,40)

plt.show()


K = 10
N,M = attributeValues_K.shape
X = np.concatenate((np.ones((attributeValues_K.shape[0],1)),attributeValues_K),1)
M = M+1

CV = model_selection.KFold(K,shuffle=True)
#values of lambda
lambdas = np.power(10.,range(-5,9))
# Initialize variables
#T = len(lambdas)
Error_train = np.empty((K,1))
Error_test = np.empty((K,1))
Error_train_rlr = np.empty((K,1))
Error_test_rlr = np.empty((K,1))
Error_train_nofeatures = np.empty((K,1))
Error_test_nofeatures = np.empty((K,1))
w_rlr = np.empty((M,K))
mu = np.empty((K, M-1))
sigma = np.empty((K, M-1))
w_noreg = np.empty((M,K))

k=0

for train_index, test_index in CV.split(X,y):
    
    # extract training and test set for current CV fold
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    internal_cross_validation = 10    
    
    opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_train, y_train, lambdas, internal_cross_validation)

    # Standardize outer fold based on training set, and save the mean and standard
    # deviations since they're part of the model (they would be needed for
    # making new predictions) - for brevity we won't always store these in the scripts
    mu[k, :] = np.mean(X_train[:, 1:], 0)
    sigma[k, :] = np.std(X_train[:, 1:], 0)
    
    X_train[:, 1:] = (X_train[:, 1:] - mu[k, :] ) / sigma[k, :] 
    X_test[:, 1:] = (X_test[:, 1:] - mu[k, :] ) / sigma[k, :] 
    
    #X_train=X_train[np.logical_not(np.isnan(X_train))]
    
    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train
    
    # Compute mean squared error without using the input data at all
    Error_train_nofeatures[k] = np.square(y_train-y_train.mean()).sum(axis=0)/y_train.shape[0]
    Error_test_nofeatures[k] = np.square(y_test-y_test.mean()).sum(axis=0)/y_test.shape[0]

    # Estimate weights for the optimal value of lambda, on entire training set
    lambdaI = opt_lambda * np.eye(M)
    lambdaI[0,0] = 0 # Do no regularize the bias term
    w_rlr[:,k] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
    # Compute mean squared error with regularization with optimal lambda
    Error_train_rlr[k] = np.square(y_train-X_train @ w_rlr[:,k]).sum(axis=0)/y_train.shape[0]
    Error_test_rlr[k] = np.square(y_test-X_test @ w_rlr[:,k]).sum(axis=0)/y_test.shape[0]

    # Estimate weights for unregularized linear regression, on entire training set
    #w_noreg[:,k] = np.linalg.solve(XtX,Xty).squeeze()
    #w_noreg[:,k] = np.linalg.lstsq(XtX,Xty,rcond=None)[0] 
    # Compute mean squared error without regularization
    #Error_train[k] = np.square(y_train-X_train @ w_noreg[:,k]).sum(axis=0)/y_train.shape[0]
    #Error_test[k] = np.square(y_test-X_test @ w_noreg[:,k]).sum(axis=0)/y_test.shape[0]
    # OR ALTERNATIVELY: you can use sklearn.linear_model module for linear regression:
    m = lm.LinearRegression().fit(X_train, y_train)
    Error_train[k] = np.square(y_train-m.predict(X_train)).sum()/y_train.shape[0]
    Error_test[k] = np.square(y_test-m.predict(X_test)).sum()/y_test.shape[0]

    # Display the results for the last cross-validation fold
    if k == K-1:
        figure(k, figsize=(12,8))
        subplot(1,2,1)
        semilogx(lambdas,mean_w_vs_lambda.T[:,1:],'.-') # Don't plot the bias term
        xlabel('Regularization factor')
        ylabel('Mean Coefficient Values')
        grid()
        # You can choose to display the legend, but it's omitted for a cleaner 
        # plot, since there are many attributes
        #legend(attributeNames[1:], loc='best')
        
        subplot(1,2,2)
        title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)))
        loglog(lambdas,train_err_vs_lambda.T,'b.-',lambdas,test_err_vs_lambda.T,'r.-')
        xlabel('Regularization factor')
        ylabel('Squared error (crossvalidation)')
        legend(['Train error','Validation error'])
        grid()
    
    # To inspect the used indices, use these print statements
    #print('Cross validation fold {0}/{1}:'.format(k+1,K))
    #print('Train indices: {0}'.format(train_index))
    #print('Test indices: {0}\n'.format(test_index))

    k+=1

show()
# Display results
print('Linear regression without feature selection:')
print('- Training error: {0}'.format(Error_train.mean()))
print('- Test error:     {0}'.format(Error_test.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test.sum())/Error_test_nofeatures.sum()))
print('Regularized linear regression:')
print('- Training error: {0}'.format(Error_train_rlr.mean()))
print('- Test error:     {0}'.format(Error_test_rlr.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train_rlr.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test_rlr.sum())/Error_test_nofeatures.sum()))

print('Weights in last fold:')
for m in range(M):
    print('{:>15} {:>15}'.format(attributeNames[m], np.round(w_rlr[m,-1],2)))

print('Ran Exercise 8.1.1')






