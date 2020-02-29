# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 11:36:52 2020

@author: jeeva
"""

import pandas as pd 
import numpy as np
glas=pd.read_csv("C:\\Users\\jeeva\\Downloads\\R assignment\\KNN\\glass.csv")

from sklearn.model_selection import train_test_split
train,test = train_test_split(glas,test_size = 0.2)
# KNN using sklearn 
# Importing Knn algorithm from sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier as KNC

# for 3 nearest neighbours 
neigh = KNC(n_neighbors= 3)
# Fitting with training data 
neigh.fit(train.iloc[:,0:9],train.iloc[:,9])
#train 
train_acc=np.mean(neigh.predict(train.iloc[:,0:9])==train.iloc[:,9])#0.8070 for 3
#train  #0.8479 for 2     #1.0 for 1 #0.7719for 5
test_acc=np.mean(neigh.predict(test.iloc[:,0:9])==test.iloc[:,9])#0.6511 for 3
#  test    0.6511 for 2      #0.7441 for 1 #0.5813for 5
avf=[]
for i in range(2,50,2):
    neigh=KNC(n_neighbors=i)
    neigh.fit(train.iloc[:,0:9],train.iloc[:,9])
    train_acc=np.mean(neigh.predict(train.iloc[:,0:9])==train.iloc[:,9])
    test_acc=np.mean(neigh.predict(test.iloc[:,0:9])==test.iloc[:,9])
    avf.append([train_acc,test_acc])
    
#help(KNC)

import matplotlib.pyplot as plt # library to do visualizations 

# train accuracy plot 
plt.plot(np.arange(3,50,2),[i[0] for i in avf],"bo-")

# test accuracy plot
plt.plot(np.arange(3,50,2),[i[1] for i in avf],"ro-")

plt.legend(["train","test"])