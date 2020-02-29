# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 19:14:16 2020

@author: jeeva
"""

import pandas as pd
import numpy as np
zooo=pd.read_csv("C:\\Users\\jeeva\\Downloads\\R assignment\\KNN\\Zoo.csv")
from sklearn.model_selection import train_test_split
train,test = train_test_split(zooo,test_size = 0.2)
# Importing Knn algorithm from sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier as KNC

neigh = KNC(n_neighbors= 4)

neigh.fit(train.iloc[:,1:17],train.iloc[:,17])

train_acc=np.mean(neigh.predict(train.iloc[:,1:17])==train.iloc[:,17])#
#train 0.987(2) 0.975(3) 0.9625(4) 0.925(5)
test_acc=np.mean(neigh.predict(test.iloc[:,1:17])==test.iloc[:,17])#
#test 0.952(2&3) 0.952(2) 0.809(5)
zcc=[]
for i in range(3,50,2):
    neigh=KNC(n_neighbors=i)
    neigh.fit(train.iloc[:,1:17],train.iloc[:,17])
    train_acc=np.mean(neigh.predict(train.iloc[:,1:17])==train.iloc[:,17])
    test_acc=np.mean(neigh.predict(test.iloc[:,1:17])==test.iloc[:,17])
    zcc.append([train_acc,test_acc])
import matplotlib.pyplot as plt

plt.plot(np.arange(3,50,2),[i[0] for i in zcc],"bo-")

# test accuracy plot
plt.plot(np.arange(3,50,2),[i[1] for i in zcc],"ro-")
plt.legend(["train","test"])