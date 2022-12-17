# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 11:08:08 2022

@author: SANJUSHA
"""

import pandas as pd
import numpy as np

# Train data
df_train=pd.read_csv("SalaryData_Train.csv")
df_train
df_train.shape
df_train.info()

# Test data
df_test=pd.read_csv("SalaryData_Test.csv")
df_test
df_test.shape
df_test.info()

# Concating
df=pd.concat([df_train,df_test],axis=0)
df.shape
df.info()
df.corr()
df.corr().to_csv('naive.csv') # There is no relation b/w independent variables
df.head()

# Outliers dectection and treating outliers
df.boxplot("age",vert=False)
Q1=np.percentile(df["age"],25)
Q3=np.percentile(df["age"],75)
IQR=Q3-Q1
LW=Q1-(2.0*IQR)
UW=Q3+(2.0*IQR)
df["age"]<LW
df[df["age"]<LW]
df[df["age"]<LW].shape
df["age"]>UW
df[df["age"]>UW]
df[df["age"]>UW].shape
df["age"]=np.where(df["age"]>UW,UW,np.where(df["age"]<LW,LW,df["age"]))

df.boxplot("educationno",vert=False)
Q1=np.percentile(df["educationno"],25)
Q3=np.percentile(df["educationno"],75)
IQR=Q3-Q1
LW=Q1-(2.0*IQR)
UW=Q3+(2.0*IQR)
df["educationno"]<LW
df[df["educationno"]<LW]
df[df["educationno"]<LW].shape
df["educationno"]>UW
df[df["educationno"]>UW]
df[df["educationno"]>UW].shape
df["educationno"]=np.where(df["educationno"]>UW,UW,np.where(df["educationno"]<LW,LW,df["educationno"]))

df.boxplot("hoursperweek",vert=False)
Q1=np.percentile(df["hoursperweek"],25)
Q3=np.percentile(df["hoursperweek"],75)
IQR=Q3-Q1
LW=Q1-(2.0*IQR)
UW=Q3+(2.0*IQR)
df["hoursperweek"]<LW
df[df["hoursperweek"]<LW]
df[df["hoursperweek"]<LW].shape
df["hoursperweek"]>UW
df[df["hoursperweek"]>UW]
df[df["hoursperweek"]>UW].shape
df["hoursperweek"]=np.where(df["hoursperweek"]>UW,UW,np.where(df["hoursperweek"]<LW,LW,df["hoursperweek"]))

df.boxplot("capitalgain",vert=False)
Q1=np.percentile(df["capitalgain"],25)
Q3=np.percentile(df["capitalgain"],75)
IQR=Q3-Q1
LW=Q1-(2.0*IQR)
UW=Q3+(2.0*IQR)
df["capitalgain"]<LW
df[df["capitalgain"]<LW]
df[df["capitalgain"]<LW].shape
df["capitalgain"]>UW
df[df["capitalgain"]>UW]
df[df["capitalgain"]>UW].shape
df["capitalgain"]=np.where(df["capitalgain"]>UW,UW,np.where(df["capitalgain"]<LW,LW,df["capitalgain"]))

df.boxplot("capitalloss",vert=False)
Q1=np.percentile(df["capitalloss"],25)
Q3=np.percentile(df["capitalloss"],75)
IQR=Q3-Q1
LW=Q1-(2.0*IQR)
UW=Q3+(2.0*IQR)
df["capitalloss"]<LW
df[df["capitalloss"]<LW]
df[df["capitalloss"]<LW].shape
df["capitalloss"]>UW
df[df["capitalloss"]>UW]
df[df["capitalloss"]>UW].shape
df["capitalloss"]=np.where(df["capitalloss"]>UW,UW,np.where(df["capitalloss"]<LW,LW,df["capitalloss"]))

# Spliting 
X=df.iloc[:,0:13]
X.columns
Y=df["Salary"]

# Standardization is not requird because naive bayes algorithm work on the principle of conditional probability
# Changing categorical into numerical
from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()

X["workclass"]=LE.fit_transform(X["workclass"])
X["education"]=LE.fit_transform(X["education"])
X["maritalstatus"]=LE.fit_transform(X["maritalstatus"])
X["occupation"]=LE.fit_transform(X["occupation"])
X["relationship"]=LE.fit_transform(X["relationship"])
X["race"]=LE.fit_transform(X["race"])
X["sex"]=LE.fit_transform(X["sex"])
X["native"]=LE.fit_transform(X["native"])
X

# Train and Test
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3)

# Model fitting
from sklearn.naive_bayes import MultinomialNB
NB=MultinomialNB()
NB.fit(X_train,Y_train) 
Y_predtrain=NB.predict(X_train)
Y_predtest=NB.predict(X_test)

from sklearn.metrics import accuracy_score
ac1=accuracy_score(Y_train,Y_predtrain)
ac2=accuracy_score(Y_test,Y_predtest)

prediction=NB.predict(np.array([[36,0,2,13,0,0,1,1,1,0,0,40,16]]))

# Inference :
# Train Accuracy=77.44% and Test Accuracy=77.15%
# Prediction = <=50K



