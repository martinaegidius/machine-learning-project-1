import touch
import pandas as pd 

df = pd.read_csv("marketing.data",sep=" ",header=None)
missingElms = df.isna().any(axis=1).sum()
print("number of missing observations: " + str(missingElms))
print("Number of observations in total: " + str(df.shape[0]))
print("Total number of valid observations: " + str(df.shape[0]-missingElms))


###describing statistics for missing data
missingElms2 = df.isna()
missingElmsSum = missingElms2.sum(axis=0).transpose() #counting sum of missing values attributewise
#ax = missingElmsSum.plot(kind='bar',rot=0)


#general statistics 
means = df[:].mean(axis=0).round().transpose()
ax2 = means.plot(kind='bar',rot=0)

summary = df[:].describe()

