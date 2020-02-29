# -*- coding: utf-8 -*-
#importing packages 
import pandas as pd #for read a dataset
fraud=pd.read_csv("C:\\Users\\jeeva\\Downloads\\R assignment\\decision tree\\Fraud_check.csv")
#change categorical column values to numerical data values
fraud.Undergrad[fraud.Undergrad=='YES']=1
fraud.Undergrad[fraud.Undergrad=='NO']=0
fraud.Urban[fraud.Urban=='YES']=1
fraud.Urban[fraud.Urban=='NO']=0

#Rename a taxable income column as Tax
fraud.rename(columns={'Taxable.Income':'Tax'},inplace=True)
#if tax less than 30000 we denote as Risky otherwise Good
fraud.Tax[fraud.Tax<=30000]='Risky'
fraud.Tax[fraud.Tax!='Risky']='Good'
#Rename a marital status column as marital
fraud.rename(columns={'Marital.Status':'Marital'},inplace=True)
fraud.Marital.unique()
#['Single', 'Divorced', 'Married']
#change marital categorical datas to numerical data
def tochn(i):
    if i=='Single':
        return 0
    if i=='Divorced':
        return 1
    if i=='Married':
        return 2
fraud['Marital']=fraud['Marital'].apply(tochn)
#create new column with input variables as first output as last
fraud1=fraud[['Undergrad','Marital','City.Population','Work.Experience','Urban','Tax']]
colnames=list(fraud1.columns)
#predict and target variables assigned
prde=colnames[0:5]
trge=colnames[5]

import numpy as np #for mathematical operation
from sklearn.model_selection import train_test_split #for spliting train and test datas
train,test=train_test_split(fraud1,test_size=0.2)

from sklearn.tree import DecisionTreeClassifier  #import decision tree classfier
model=DecisionTreeClassifier(criterion="gini") #DecisionTreeClassifier with criterion gini index
model.fit(train[prde],train[trge])#fitting with train predict and train target
pred=model.predict(test[prde])#predict a model with test input

pd.Series(pred).value_counts()#total Good And Risky conditions
pd.crosstab(test[trge],pred)#cross tab function for mismatch
np.mean(pred==test.Tax)
# accuracy -- 0.6083333333333333
model=DecisionTreeClassifier(criterion="entropy")#DecisionTreeClassifier with criterion entropy index
model.fit(train[prde],train[trge])#fitting with train input and train target
pred=model.predict(test[prde])#predict a model with test input

pd.Series(pred).value_counts()#total Good And Risky conditions
pd.crosstab(test[trge],pred)#cross tab function for mismatch
np.mean(pred==test.Tax)
# Accuracy   ---  0.7

#the criterion entropy gives high accuaracy so we use this criterion for this analysis