# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 18:16:33 2020

@author: Dinesh
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

os.chdir("D:\DATASETS_MACHINE_LEARNING_PRACTICE")

df=pd.read_csv("Placement_Data.csv",index_col=None,usecols=range(1,15))
# df.head()
# df.info()

for col_name in df.columns:
    if(df[col_name].dtype == 'object'):
        df[col_name]= df[col_name].astype('category')
        df[col_name] = df[col_name].cat.codes

df['salary']=df['salary'].fillna(0)
per_col=['ssc_p','hsc_p','degree_p','etest_p','mba_p']
for i in per_col:
    df[i]=df[i]/100
    
###  MULTIPLE LINEAR REGRESSION    ###############

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

x=df.iloc[:,: -1]
y=df.iloc[:,13]

y=y/100


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=0)

###  MULTIPLE LINEAR REGRESSION    ###############
# from sklearn.linear_model import LinearRegression
# reg=LinearRegression()
# reg.fit(X_train,Y_train)
# y_pred=reg.predict(X_test)

# from sklearn.metrics import r2_score
# score=r2_score(Y_test,y_pred)

# print(score*100)


###  SVM REGRESSION    ###############
# from sklearn.svm import SVR
# from sklearn.model_selection import GridSearchCV
# regres=SVR(C=10,kernel='rbf')
# regres.fit(X_train,Y_train)
# param = {'kernel' : ('linear', 'poly', 'rbf', 'sigmoid'),'C' : [1,5,10,15,20,25],'degree' : [3,8],'coef0' : [0.01,0.001,10,0.5],'gamma' : ('auto','scale')}

# modelsvr = SVR()

# grids = GridSearchCV(modelsvr,param,cv=5)

# grids.fit(X_train,Y_train)


# y_pred1=regres.predict(X_test)
# y_pred2=grids.predict(X_test)
# from sklearn.metrics import r2_score
# score1=r2_score(Y_test,y_pred1)
# score2=r2_score(Y_test,y_pred2)
# print("Accuracy for SVR model")
# print(score1*100)
# print(score2*100)#accuracy=84



############ Decision tree Regressor  ####################
# from sklearn.tree import DecisionTreeRegressor
# regressor = DecisionTreeRegressor(random_state = 100)  
  
# # fit the regressor with X and Y data 
# regressor.fit(X_train,Y_train)

# y_pred3=regressor.predict(X_test) 
# score3=r2_score(Y_test, y_pred3)
# print("Decision tree regressor")
#print(score3*100) #accuracy=24



#Lasso Regression
############################################################################
from sklearn.linear_model import Lasso

#Initializing the Lasso Regressor with Normalization Factor as True
lasso_reg = Lasso(normalize=True)

#Fitting the Training data to the Lasso regressor
lasso_reg.fit(X_train,Y_train)

#Predicting for X_test
y_pred_lass =lasso_reg.predict(X_test)

print("\n\nLasso SCORE : ", r2_score(y_pred_lass, Y_test)*100)



######## RIDGE RESGRESSION ########################
# from sklearn.linear_model import Ridge
# mod1 = Ridge()
# mod1.fit(X_train, Y_train)
# pred1=mod1.predict(X_test)

# print("\n\nRidge SCORE : ", r2_score(pred1, Y_test)*100)




# from sklearn.linear_model import ElasticNetCV
# mod = ElasticNetCV()
# mod.fit(X_train, Y_train)

# pred=mod.predict(X_test)

# print("\n\nELASTIC NET SCORE : ", r2_score(pred, Y_test)*100)




