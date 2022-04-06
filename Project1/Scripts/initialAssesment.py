import touch
import pandas as pd 
import seaborn as sns

df = pd.read_csv("/home/max/Documents/s194119/Machine_learning/Project1/Data/marketing.data",sep=" ",header=None)
missingElms = df.isna().any(axis=1).sum()
print("number of missing observations: " + str(missingElms))
print("Number of observations in total: " + str(df.shape[0]))
print("Total number of valid observations: " + str(df.shape[0]-missingElms))


###describing statistics for missing data
missingElms2 = df.isna()
missingElmsSum = missingElms2.sum(axis=0)
column_names = ["Attributes","Number of missing entries"]
df2 = pd.DataFrame(columns=column_names)
df2["Number of missing entries"] = missingElmsSum[:]
df2["Attribute"] = pd.Series(range(1,15))
sns.set_style("whitegrid")
g = sns.barplot(x = df2["Attribute"], y = df2["Number of missing entries"],ci=None)

summary = df[:].describe()

