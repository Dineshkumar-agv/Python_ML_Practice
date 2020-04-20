# -*- coding: utf-8 -*-
"""
@author: DINESH_AGV
"""
import numpy as np
import pandas as pd
import matplotlib as mp
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

a=os.getcwd()
print(a)
os.chdir('D:\MACHINE_LEARNING_PRACTICE')
a=os.getcwd()
print(a)

df=pd.read_csv("loans.csv")
#df=pd.read_csv("winequality-red.csv")

#identifying and imputing missing data
f=df.isnull().sum()

df['days.with.cr.line']=df['days.with.cr.line'].fillna(np.mean(df['days.with.cr.line']))
df['log.annual.inc']=df['log.annual.inc'].fillna(np.mean(df['log.annual.inc']))
df['revol.util']=df['revol.util'].fillna(np.mean(df['revol.util']))
df['inq.last.6mths']=df['inq.last.6mths'].fillna(np.mean(df['inq.last.6mths']))
df['delinq.2yrs']=df['delinq.2yrs'].fillna(np.mean(df['delinq.2yrs']))
df['pub.rec']=df['pub.rec'].fillna(np.mean(df['pub.rec']))

#converting categorical variables into continuous variables
df['purpose']=df['purpose'].astype('category')

cat_columns=df.select_dtypes(['category']).columns
df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)

    
# df['quality']=df['quality'].replace(df['quality'].apply(lambda x:True if(x>5)else False))

#separating input and output data
x=df.iloc[:,: -1].values
y=df.iloc[:,13].values

X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.25,random_state=50)

clf_entropy=DecisionTreeClassifier(criterion='gini',random_state=50,max_leaf_nodes=5,max_depth=2,min_samples_leaf=50)

# X_train=X_train.reshape(-1,1)

clf_entropy.fit(X_train,Y_train)

Y_pred=clf_entropy.predict(X_test)


print('Accuracy Score is:::',accuracy_score(Y_test,Y_pred)*100)

df1=df
del df1['not.fully.paid']
features=list(df1.columns)
from IPython.display import Image
from sklearn.externals.six import StringIO
import pydotplus
# dot_data=StringIO()
# features=list(df1.columns)
# tree.export_graphviz(clf_entropy,out_file=dot_data,rounded=True,filled=True,feature_names=features,impurity=False)
# graph=pydotplus.graph_from_dot_data(dot_data.getvalue())
# Image(graph.create_png())
from dtreeplt import dtreeplt

dtree = dtreeplt(model=df1,feature_names=features)
fig = dtree.view()



# df1=df.reshape(-1,1)
# dot_file=open("pt.dot",'w')
# sd=tree.export_graphviz(clf_entropy,out_file=dot_file,feature_names=df1.columns)