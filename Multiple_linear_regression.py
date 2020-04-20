# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 02:45:28 2020

@author: DELL
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

dataset=pd.read_csv('Data.csv')
# sns.heatmap(dataset.isnull(),yticklabels=False,cbar=False,cmap='ocean')
df=dataset

#converting categorical data into numerical data
for col_name in df.columns:
    if(df[col_name].dtype == 'object'):
        df[col_name]= df[col_name].astype('category')
        df[col_name] = df[col_name].cat.codes
        
       
#identifying and imputing missing data
f=df.isnull().sum() # not necessary but just to visualize

for i in df:
    df[i]=df[i].fillna(np.mean(df[i])) # this can be implemented in the above loop
        
print('first part')

# splitting data into train and test subsets
x=df.iloc[:,: -1].values
y=df.iloc[:,8].values
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=0)

import statsmodels.api as sm
model=sm.OLS(Y_train,X_train).fit()

print(model.summary())

Y_pred=model.predict(X_test)


print('MAE:::',metrics.mean_absolute_error(Y_test,Y_pred))

print('MSE:::',metrics.mean_squared_error(Y_test,Y_pred))

print('RMSE:::',np.sqrt(metrics.mean_absolute_error(Y_test,Y_pred)))






