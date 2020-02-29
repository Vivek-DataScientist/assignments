
import pandas as pd#import pandas module as pd 
import matplotlib.pyplot as plt  #import plotting function
#read a dataset
comp=pd.read_csv("C:\\Users\\jeeva\\Downloads\\R assignment\\decision tree\\Company_Data.csv")
#change bad=0 medium=1 good=2 in shelveloc column
def tochange(i):
    if i=='Bad':
        return 0
    if i=='Medium':
        return 1
    if i=='Good':
        return 2
comp['ShelveLoc']=comp['ShelveLoc'].apply(tochange)

#convert salesRanges values to some unique values 
def toconv(x):
    if x=='[0-1.0]':
        return 0
    if x=='[1-2.0]':
        return 1
    if x=='[2-3.0]':
        return 2
    if x=='[3-4.0]':
        return 3
    if x=='[4-5.0]':
        return 4
    if x=='[5-6.0]':
        return 5
    if x=='[6-7.0]':
        return 6
    if x=='[7-8.0]':
        return 7
    if x=='[8-9.0]':
        return 8
    if x=='[9-10.0]':
        return 9
    if x=='[10-11.0]':
        return 10
    if x=='[11-12.0]':
        return 11
    if x=='[12-13.0]':
        return 12
    if x=='[13-14.0]':
        return 13
    if x=='[14-15.0]':
        return 14
    if x=='[15-16.0]':
        return 15
    if x=='[16-17.0]':
        return 16
    
#Change categorical to numeical
comp.Urban[comp.Urban=='Yes']=1
comp.Urban[comp.Urban=='No']=0
comp.US[comp.US=='Yes']=1
comp.US[comp.US=='No']=0

comp.Sales.max()
#they give sales ranges
Sales_ranges=["[{0}-{1}]".format(Sales,Sales+1.0)for Sales in range(0,17,1)]
Sales_ranges#we identify sales ranges
count_Sales_ranges=len(Sales_ranges)#count sales ranges

#count the sales ranges  values  ex:[9-10.0]=53 counts [16-17.0]=2counts
comp['Salesranges_c']=pd.cut(x=comp['Sales'],bins=count_Sales_ranges,labels=Sales_ranges)
Sales_len=comp['Salesranges_c'].value_counts()

comp_range=pd.DataFrame(Sales_len).reset_index()
#create a column with ranges and count values of sales_ranges
comp_range.columns=['Salesranges_c','count']
comp_range

#plotting sales ranges values
plt.bar(comp_range['Salesranges_c'],comp_range['count'])
plt.show()

#create column to change particular ranges to numerical for identifying
comp['Sales_out']=comp['Salesranges_c'].apply(toconv)#call toconv function
#create column change numerical data to categorical data
comp['sale']=comp['Sales_out'].replace([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q'])
#########Converted to Categorical datas#####

import numpy as np#for mathematical operation
from sklearn.model_selection import train_test_split #split a train and test data
#inputs are first and output as last in different dataframe
comp1=comp[['CompPrice','Income','Advertising','Population','Price','ShelveLoc','Age','Education','Urban','US','sale']]
colnames=list(comp1.columns)
pred=colnames[0:10] #predict a input columns
targ=colnames[10] #output as target

#split the data 
train,test = train_test_split(comp1,test_size = 0.2)

#import decision tree
from sklearn.tree import  DecisionTreeClassifier

#model prepared by decision tree classifier 
model = DecisionTreeClassifier(criterion = 'entropy')
model.fit(train[pred],train[targ]) #fitt values to input to output
preds = model.predict(test[pred]) #predict a model with test input
pd.Series(preds).value_counts() #count predictions

pd.crosstab(test[targ],preds)#cross tab for mismatch function
np.mean(preds==test.sale) # 0.075 for gini: 0.1625 for entropy
#I    13 #G    13 #J    11
#E     8 #L     8 #F     6
#K     4 #N     3 #A     3
#C     3 #H     2 #M     2
#D     2 #B     2