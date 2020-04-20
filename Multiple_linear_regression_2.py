# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 20:44:04 2020

@author: Dinesh
"""

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn import metrics
a=os.getcwd()
print(a)
os.chdir('D:\MACHINE_LEARNING_PRACTICE')
a=os.getcwd()
print(a)

dataset=pd.read_csv('50_Startups.csv')

sns.heatmap(dataset.isnull(),yticklabels=False,cbar=False,cmap='ocean')
df=dataset

f=df.isnull().sum() # not necessary but just to visualize

for i in df:
    if(df[i].dtype != 'object'):
        df[i]=df[i].fillna(np.mean(df[i])) # this can be implemented in the above loop
        
print('first part')

fds=df.isnull().sum() 
#converting categorical data into numerical data
# for col_name in df.columns:
#     if(df[col_name].dtype == 'object'):
#         df[col_name]= df[col_name].astype('category')
#         df[col_name] = df[col_name].cat.codes
        




# # splitting data into train and test subsets
x=df.iloc[:,: -1]
y=df.iloc[:,4]


for col_name in x.columns:
    print('Column name is::',col_name)



for col_name in x.columns:
       if(df[col_name].dtype == 'object'):
           labels=pd.get_dummies(x[col_name],drop_first=True)
           x=x.drop(col_name,axis=1)
           x=pd.concat([x,labels],axis=1)


# for i in x:
#     x[i]=x[i].fillna(np.mean(x[i])) # this can be implemented in the above loop

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=22)

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,Y_train)
y_pred=reg.predict(X_test)

from sklearn.metrics import r2_score
score=r2_score(Y_test,y_pred)

print('Accuracy is::',score*100)


df1 = pd.DataFrame({'Actual': Y_test, 'Predicted': y_pred.flatten()})


df2 = df1.head(25)
df2.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()



