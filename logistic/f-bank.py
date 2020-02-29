# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 11:18:42 2020

@author: jeeva
"""
#import packages
import pandas as pd #import pandas module for reading dataset
import matplotlib.pyplot as plt#for plotting function
from sklearn.linear_model import LogisticRegression #importing logistic regression function
from sklearn.model_selection import train_test_split #split the data into train and test data
#read a dataset
cc=pd.read_csv("C:\\Users\\jeeva\\Downloads\\R assignment\\logistic\\a.csv")
v1=cc.dropna()#if null value occur then droped it & stored in v1

#categorical_cols (yes&no) ="default,housing,loan,y"
#numerical_cols="age,balance,day,duration,campaign,pdays,previous"
#categorical data='job','marital','education','contact','month','poutcome'

count_nodeposit= len(v1[v1['y']=='no'])#39922 no values in Output
count_deposit=len(v1[v1['y']=='yes'])#5289 yes values in Output
pct_count_nodeposit=count_nodeposit/(count_nodeposit+count_deposit)
print("percentage of no deposit",pct_count_nodeposit*100) #No deposit =88.30%
pct_countdeposit=count_deposit/(count_nodeposit+count_deposit)
print("percentage of deposit",pct_countdeposit*100) #deposit =11.69%
    ###### CATEGORICAL MEANS ############
m1=v1.groupby('y').mean() #based on categorical other attribute mean
job=v1.groupby('job').mean() #based on categorical other attribute mean
marital=v1.groupby('marital').mean() #based on categorical other attribute mean
    ###### CATEGORICAL vs Output plots ############
    #plotting different different plots#####
table=pd.crosstab(v1.marital,v1.y)
pd.crosstab(v1.marital,v1.y).plot(kind='bar')
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=False)
plt.title="marital vs output"
plt.x="marital"
plt.y="output"

pd.crosstab(v1.month,v1.y).plot(kind='bar')
####################### DUMMY VARIABLES FOR categorical data############

col_dumm= ['job','marital','education','default','housing','loan','contact','month','poutcome']
for var in col_dumm:
    cat_lis='var'+'_'+var
    cat_lis= pd.get_dummies(v1[var], prefix=var)
    data1=v1.join(cat_lis)
    v1=data1 #adding dummy datas to v1

data_vars=v1.columns.values.tolist()
to_keep=[i for i in data_vars if i not in cat_lis]
data_final=v1[to_keep]
data_final.columns.values
#deleted categorical datas after added dummy columns
data_final.drop(['job','marital','education','default','housing','loan','contact','month','poutcome'],inplace=True,axis=1)

###With our training data created, I’ll up-sample the no-subscription using the SMOTE algorithm
###(Synthetic Minority Oversampling Technique)
X = data_final.loc[:, data_final.columns != 'y']#read all values except output
y = data_final.loc[:, data_final.columns == 'y']#read output value

from imblearn.over_sampling import SMOTE #importing Smote 
os = SMOTE(random_state=0)
#split datas to train and test values of X and y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
columns = X_train.columns

os_data_X,os_data_y=os.fit_sample(X_train, y_train)#fitting the values of train datas
os_data_X = pd.DataFrame(data=os_data_X,columns=columns )
os_data_y= pd.DataFrame(data=os_data_y,columns=['y'])
# we can Check the numbers of our data
print("length of oversampled data is ",len(os_data_X))#55906
print("Number of no subscription in oversampled data",len(os_data_y[os_data_y['y']=='no']))#27953
print("Number of subscription",len(os_data_y[os_data_y['y']=='yes']))#27593
print("Proportion of no subscription data in oversampled data is ",len(os_data_y[os_data_y['y']=='no'])/len(os_data_X))#0.5
print("Proportion of subscription data in oversampled data is ",len(os_data_y[os_data_y['y']=='yes'])/len(os_data_X))#0.5


logreg = LogisticRegression()#logistic regression
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)#predict y as x_test

from sklearn.metrics import confusion_matrix #import confusion matrix
confusion_matrix = confusion_matrix(y_test, y_pred) #confusion matrix for output test and prediction
print(confusion_matrix)
#[11718   251]
#[ 1289   306]]

from sklearn.metrics import classification_report#for reporting
print(classification_report(y_test, y_pred))
   #           precision    recall  f1-score   support
   #       no       0.90      0.98      0.94     11969
  #       yes       0.55      0.19      0.28      1595
  #  accuracy                           0.89     13564
  # macro avg       0.73      0.59      0.61     13564
#weighted avg       0.86      0.89      0.86     13564