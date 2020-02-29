#import dataset
import pandas as pd
#read a dataset
fraud=pd.read_csv("C:\\Users\\jeeva\\Downloads\\R assignment\\random forest\\Fraud_check.csv")
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
X=fraud1[prde]
Y=fraud1[trge]
from sklearn.ensemble import RandomForestClassifier#random forest classifier
rf = RandomForestClassifier(n_jobs=2,oob_score=True,n_estimators=750,criterion="entropy")
#N estimators 750 for tree better result
#### Attributes that comes along with RandomForest function
rf.fit(X,Y) # Fitting RandomForestClassifier model from sklearn.ensemble 
rf.estimators_ # 
rf.classes_ # class labels (output)
rf.n_classes_ # Number of levels in class labels 
rf.n_features_  # Number of input features in model 8 here.

rf.n_outputs_ # Number of outputs when fit performed

rf.predict(X)

fraud1['rf_pred'] = rf.predict(X)#create column with prediction

from sklearn.metrics import confusion_matrix
confusion_matrix(fraud1['Tax'],fraud1['rf_pred']) # Confusion matrix

pd.crosstab(fraud1['Tax'],fraud1['rf_pred'])
#nesti=750 good 476 Risky 124 =600 :::::100% accuracy
rf1 = RandomForestClassifier(n_jobs=2,oob_score=True,n_estimators=60,criterion="gini")

#### Attributes that comes along with RandomForest function
rf1.fit(X,Y)
rf1.predict(X)
fraud1['rf1_pred'] = rf1.predict(X)

from sklearn.metrics import confusion_matrix
confusion_matrix(fraud1['Tax'],fraud1['rf1_pred']) # Confusion matrix

pd.crosstab(fraud1['Tax'],fraud1['rf1_pred'])
#nesti=600,300 good 476 Risky 124 =600 :::::100% accuracy
#nesti=60 good 476 Risky 121   goodRisky=3 =600 :::::97% accuracy