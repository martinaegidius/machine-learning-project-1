# exercise 4.1.3

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy import stats

mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["#0B2436","#C9A3A4","#D1B899","#BBBBBE","#AEB5AC","#858686","#ADCDEC","#B6BDCA"]) 

means_error = []
std_error = []
numPoints = []

for i in range(100):
# Number of samples
    N = 100+i*1000
    numPoints.append(N)
    
    # Mean
    mu = 17
    
    # Standard deviation
    s = 2
    
    # Number of bins in histogram
    nbins = 20
    
    # Generate samples from the Normal distribution
    X = np.random.normal(mu,s,N).T 
    # or equally:
    #X = np.random.randn(N).T * s + mu
    
    # Plot the histogram
    #f = figure()
    #title('Normal distribution')
    #hist(X, bins=nbins, density=True)
    
    # Over the histogram, plot the theoretical probability distribution function:
    #x = np.linspace(X.min(), X.max(), 1000)
    #pdf = stats.norm.pdf(x,loc=17,scale=2)
    #plot(x,pdf,'.',color='red')
    
    # Compute empirical mean and standard deviation
    mu_ = X.mean()
    s_ = X.std(ddof=1)
    means_error.append(abs(mu-mu_))
    std_error.append(abs(s-s_))

#print("Theoretical mean: ", mu)
#print("Theoretical std.dev.: ", s)
#print("Empirical mean: ", mu_)
#print("Empirical std.dev.: ", s_)

#show()

#print('Ran Exercise 4.1.3')

fig = plt.plot(numPoints,means_error)
plt.plot(numPoints,std_error)
plt.legend(["means","std"])
plt.xlabel("Number of random samples")
plt.ylabel("Absolute error")
plt.show()