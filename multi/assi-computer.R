#read computer dataset
cc<-read.csv(file.choose())
View(cc)#viewing dataset
#change categorical data to numeric data
cc$cd=as.numeric(factor(cc$cd))
cc$multi=as.numeric(factor(cc$multi))
cc$premium=as.numeric(factor(cc$premium))
#remove a first column 1st column contains serial number
cc<-cc[,-1]
View(cc)
attach(cc)#attach the dataset after removing 1st column
#model preparation Price is output other columns are inputs.
modela=lm(price~speed+hd+ram+screen+cd+multi+premium+ads+trend)
summary(modela)##P-Value<0.05 R-squared:  0.7756,	Adjusted R-squared:  0.7752 

#correlation analysis for the dataset
cor(cc)

#importing package corpcor
library(corpcor)

#Partial correlation analysis with correlation of the dataset
cor2pcor(cor(cc)) #pairwise partial correlation coefficients
#importing car package
library(car)
#influence factors its identify outlier of the dataset 
influence.measures(modela)
influenceIndexPlot(modela,id.n=3)
influencePlot(modela)
#identified 1441,1701 in plot it has outlier if i deleted then R & Adjusted R square will improve 
#find the model after deleting outliers of dataset
modelaa=lm(price~speed+hd+ram+screen+cd+multi+premium+ads+trend,data=cc[-1441,])#remove 1441th row of dataset
modelaa=lm(price~speed+hd+ram+screen+cd+multi+premium+ads+trend,data=cc[-1701,])#remove 1701th row of dataset
summary(modelaa)# P value< 0.05 R-squared:  0.7766,	Adjusted R-squared:  0.7763
#AIC method to find less models
stepAIC(modela)
#not occurs p values as greater than 0.05.so less models not produced for this dataset.

