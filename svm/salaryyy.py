# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 10:38:30 2020

@author: jeeva
"""
#import packages 
import pandas as pd #pandas for read a dataset
import numpy as np #for mathematical operation
train_sal=pd.read_csv("C:\\Users\\jeeva\\Downloads\\R assignment\\svm\\SalaryData_Train(1).csv")
test_sal=pd.read_csv("C:\\Users\\jeeva\\Downloads\\R assignment\\svm\\SalaryData_Test(1).csv")

#labelEncoder for change a categorical datas to numerical datas
from sklearn.preprocessing import LabelEncoder
number=LabelEncoder()
train_sal['workclass']=number.fit_transform(train_sal['workclass'].astype('str'))
train_sal['education']=number.fit_transform(train_sal['education'].astype('str'))
train_sal['maritalstatus']=number.fit_transform(train_sal['maritalstatus'].astype('str'))
train_sal['occupation']=number.fit_transform(train_sal['occupation'].astype('str'))
train_sal['relationship']=number.fit_transform(train_sal['relationship'].astype('str'))
train_sal['race']=number.fit_transform(train_sal['race'].astype('str'))
train_sal['sex']=number.fit_transform(train_sal['sex'].astype('str'))
train_sal['native']=number.fit_transform(train_sal['native'].astype('str'))
#if salary >50k we provide greater else lesser as categorical data in new column
train_sal.loc[train_sal['Salary']==' >50K','highorlow']='greater'
train_sal.loc[train_sal['Salary']==' <=50K','highorlow']='lesser'

#change a test categorical datas to numerical datas
test_sal['workclass']=number.fit_transform(test_sal['workclass'].astype('str'))
test_sal['education']=number.fit_transform(test_sal['education'].astype('str'))
test_sal['maritalstatus']=number.fit_transform(test_sal['maritalstatus'].astype('str'))
test_sal['occupation']=number.fit_transform(test_sal['occupation'].astype('str'))
test_sal['relationship']=number.fit_transform(test_sal['relationship'].astype('str'))
test_sal['race']=number.fit_transform(test_sal['race'].astype('str'))
test_sal['sex']=number.fit_transform(test_sal['sex'].astype('str'))
test_sal['native']=number.fit_transform(test_sal['native'].astype('str'))
#if salary >50k we provide greater else lesser as categorical data in new column
test_sal.loc[test_sal['Salary']==' >50K','highorlow']='greater'
test_sal.loc[test_sal['Salary']==' <=50K','highorlow']='lesser'
# x=='<=50k'  ==lesser               x=='>50k' ==greater

#split a input and output variable
train_X = train_sal.iloc[:,0:13]
train_y = train_sal.iloc[:,14]
test_X  = test_sal.iloc[:,0:13]
test_y  = test_sal.iloc[:,14]

# Create SVM classification object 
# 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
#model prepared by kernel then fitting to train and test of input datas
#predict model data with test data
#take mean of predict model with test output

# kernel = poly
#############
from sklearn.svm import SVC
model_poly = SVC(kernel = "poly")
model_poly.fit(train_X,train_y)
pred_test_poly = model_poly.predict(test_X)
np.mean(pred_test_poly==test_y) 
#0.7795484727755644

#model prepared by rbf kernel
model_rbf = SVC(kernel = "rbf")
model_rbf.fit(train_X,train_y)
pred_test_rbf = model_rbf.predict(test_X)
np.mean(pred_test_rbf==test_y) 
#0.7964143426294821

#model prepared by sigmoid kernel
model_sig = SVC(kernel = "sigmoid")
model_sig.fit(train_X,train_y)
pred_test_sig = model_sig.predict(test_X)
np.mean(pred_test_sig==test_y) 
#0.7567729083665339

#model prepared by precomputed kernel
model_pre = SVC(kernel = "precomputed")
model_pre.fit(train_X,train_y)
pred_test_pre = model_pre.predict(test_X)
np.mean(pred_test_pre==test_y) 
#precomputed only for square matrix

## RBF kernel gives high accuracy of test datas ...so we use this technique ###