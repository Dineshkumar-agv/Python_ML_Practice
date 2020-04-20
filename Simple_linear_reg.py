# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 02:04:02 2020

@author: DELL
"""
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
a=os.getcwd()
print(a)
os.chdir('D:\MACHINE_LEARNING_PRACTICE')
a=os.getcwd()
print(a)

dataset=pd.read_csv('Summary of Weather.csv')
# sns.heatmap(dataset.isnull(),yticklabels=False,cbar=False,cmap='ocean')
df=dataset
sd=df['']
# df['Parking_type']=df['Parking_type'].astype('category')
# df['City_type']=df['City_type'].astype('category')
# cat_columns=df.select_dtypes(['category']).columns

# df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)


# nan_values = df.isna()
# nan_columns = nan_values.any()

# columns_with_nan = df.columns[nan_columns].tolist()

# print(columns_with_nan)

# df=df.replace(df['Builtup_area'][27],np.nan)
# print(df['Builtup_area'][27])
# c=np.mean(df['Builtup_area'])
# print(c)
# d=np.median(df['Builtup_area'])
# print(d)
# e=stats.mode(df['Builtup_area'])
# print(e)

# df1=df
# #knn impute missing values python
# from sklearn.impute import KNNImputer
# imputer = KNNImputer(n_neighbors=8, weights="uniform")
# df1=imputer.fit_transform(df1)

# df['Builtup_area']=df['Builtup_area'].fillna(np.mean(df['Builtup_area']))
# df['Taxi_dist']=df['Taxi_dist'].fillna(np.mean(df['Taxi_dist']))
# df['Market_dist']=df['Market_dist'].fillna(np.mean(df['Market_dist']))
# df['Hospital_dist']=df['Hospital_dist'].fillna(np.mean(df['Hospital_dist']))
# df['Carpet_area']=df['Carpet_area'].fillna(np.mean(df['Carpet_area']))




# x=df.iloc[:,: -1].values
# y=df.iloc[:,8].values

# Parking_type=pd.get_dummies(df['Parking_type'],drop_first=True)
# df=df.drop('Parking_type',axis=1)
# df=pd.concat([df,Parking_type],axis=1)

# City_type=pd.get_dummies(df['City_type'],drop_first=True)
# df=df.drop('City_type',axis=1)
# df=pd.concat([df,City_type],axis=1)


# from sklearn.model_selection import train_test_split
# X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=0)
# from sklearn.linear_model import LinearRegression
# regressor=LinearRegression()
# regressor.fit(X_train, Y_train)



#Y_pred=regressor.predict(Y_test)
