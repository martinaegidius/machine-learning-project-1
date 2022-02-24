
from niko_loader import X, attributeNames
#from matplotlib.pyplot import boxplot, xticks, ylabel, title, show
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

#attributeNames

df = pd.DataFrame(X,columns=attributeNames)

#boxplot(X[:,:3])
#xticks(range(1,4),attributeNames[:3])
#ylabel('Class')
#title('Marketing data')
#show()


#sns.catplot(x = "Attribute", y="Categorical group", kind = "boxen",data = X[:,:3])
#Maritial status', 'Age',
#       'Education', 'Occupation',
#       'Time lived in the San Fran./Oakland/San Jose area',
#       'Dual incomes (if married)', 'Persons in your household',
#       'Persons in your household under 18', 'Householder status',
#       'Type of home', 'Ethnic Classification',
#       'Language spoken most often in home'
#dfT = df.transpose()
#g = sns.catplot(kind="box", data=dfT)
#g.set_xticklabels(rotation=30) 

sns.set_style("whitegrid")
g = sns.catplot(kind="boxen", data=df[["Dual incomes (if married)","Householder status","Language spoken most often in home"]],orient="h",aspect=2)
#fig = g.get_figure()
g.savefig("boxen/dual.png") 

sns.set_style("whitegrid")
g = sns.catplot(kind="box", data=df[["Dual incomes (if married)","Householder status","Language spoken most often in home"]],orient="h",aspect=2)
#fig = g.get_figure()
g.savefig("box/dual.png") 

#g.set_yticklabels(rotation=30) 
#g.set_xticklabels(g.get_xticklabels(), rotation=30)