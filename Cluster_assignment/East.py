
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.cluster import	KMeans
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import linkage 
import scipy.cluster.hierarchy as sch
mm=pd.read_csv("C:\\Users\\jeeva\\Downloads\\R assignment\\Cluster_assignment\\modified2.csv")
#Normalise the datas 
def norm_func(i):
    x = (i-i.min())	/	(i.max()	-	i.min())
    return (x)
dff_norm = norm_func(mm.iloc[:,1:])
dff_norm.describe()
#### IMPORTING LINKAGE FUNCTION ##############
zz = linkage(dff_norm, method="complete",metric="euclidean")
#help(linkage)
############## Plot the linkage plot(means smaller distance between of points)#####
plt.figure(figsize=(20, 5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Dist')
sch.dendrogram(
    zz,
    leaf_rotation=0.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()
########## for my analysis for Eastwestairlines i take clusters level as 10 because the given dataset has verylarge ####
from sklearn.cluster import	AgglomerativeClustering 
hh_complete	=	AgglomerativeClustering(n_clusters=10,linkage='complete',affinity = "euclidean").fit(dff_norm) 

hh_complete.labels_
cluster_labelss=pd.Series(hh_complete.labels_)

mm['clust']=cluster_labelss # creating a  new column and assigning it to new column 
mm1 = mm.iloc[:,[12,0,1,2,3,4,5,6,7,8,9,10,11]]
bb=mm1.head(10)

############################## K-Means Clustering  ##############################
model2 = KMeans(n_clusters=10).fit(mm)
model2.labels_
model2.cluster_centers_
kk = list(range(2,15))
kk
TWSS = [] # variable for storing total within sum of squares for each kmeans 
for ii in kk:
    kmeans = KMeans(n_clusters = ii)
    kmeans.fit(dff_norm)
    WSS = [] # variable for storing within sum of squares for each cluster 
    for jj in range(ii):
        WSS.append(sum(cdist(dff_norm.iloc[kmeans.labels_==jj,:],kmeans.cluster_centers_[jj].reshape(1,dff_norm.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))
    
plt.plot(kk,TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS");plt.xticks(kk)


model=KMeans(n_clusters=10) 
model.fit(dff_norm)

model.labels_ # getting the labels of clusters assigned to each row 
me=pd.Series(model.labels_)  # converting numpy array into pandas series object 
mm['clust']=me
mm.head(10) # creating a  new column and assigning it to new column 
mm2=mm.iloc[:,[12,0,1,2,3,4,5,6,7,8,9,10,11]]
cc=mm2.head(10)


